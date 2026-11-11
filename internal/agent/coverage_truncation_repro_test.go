// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package agent

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/alibaba/open-code-review/internal/config/template"
	"github.com/alibaba/open-code-review/internal/llm"
	"github.com/alibaba/open-code-review/internal/model"
	"github.com/alibaba/open-code-review/internal/session"
	"github.com/alibaba/open-code-review/internal/tool"
)

// scriptedClient answers request n with script[n-1], repeating the last entry
// once the script runs out.
type scriptedClient struct {
	script []func() *llm.ChatResponse
	calls  int64
}

func (c *scriptedClient) CompletionsWithCtx(_ context.Context, _ llm.ChatRequest) (*llm.ChatResponse, error) {
	n := int(atomic.AddInt64(&c.calls, 1))
	if n > len(c.script) {
		n = len(c.script)
	}
	return c.script[n-1](), nil
}

func commentTurn(path string, withDone bool, tokens int64) func() *llm.ChatResponse {
	return func() *llm.ChatResponse {
		calls := []llm.ToolCall{{ID: "c", Type: "function", Function: llm.FunctionCall{
			Name:      "code_comment",
			Arguments: `{"comments":[{"path":"` + path + `","content":"first of several findings"}]}`,
		}}}
		if withDone {
			calls = append(calls, llm.ToolCall{ID: "d", Type: "function", Function: llm.FunctionCall{Name: "task_done", Arguments: "{}"}})
		}
		return &llm.ChatResponse{
			Choices: []llm.Choice{{Message: llm.ResponseMessage{Role: "assistant", ToolCalls: calls}, FinishReason: "tool_calls"}},
			Model:   "fake",
			Usage:   &llm.UsageInfo{PromptTokens: tokens, TotalTokens: tokens},
		}
	}
}

func emptyTurn(tokens int64) func() *llm.ChatResponse {
	return func() *llm.ChatResponse {
		content := ""
		return &llm.ChatResponse{
			Choices: []llm.Choice{{Message: llm.ResponseMessage{Role: "assistant", Content: &content}}},
			Model:   "fake",
			Usage:   &llm.UsageInfo{PromptTokens: tokens, TotalTokens: tokens},
		}
	}
}

func newTruncationReproAgent(t *testing.T, diffs []model.Diff, client llm.LLMClient, tpl template.Template, budget int64) *Agent {
	t.Helper()
	setTestHome(t, t.TempDir())
	repoDir := t.TempDir()
	sh := session.New(repoDir, "feature", "fake", session.SessionOptions{
		ReviewMode: session.ReviewModeRange,
		DiffFrom:   "main",
		DiffTo:     "feature",
		Operation:  session.OperationReview,
	})
	t.Cleanup(func() { _ = sh.Finalize() })
	collector := tool.NewCommentCollector()
	reg := tool.NewRegistry()
	reg.Register(&tool.CodeCommentProvider{Collector: collector})
	a := New(Args{
		RepoDir:          repoDir,
		From:             "main",
		To:               "feature",
		ReviewMode:       session.ReviewModeRange,
		LLMClient:        client,
		Model:            "fake",
		Session:          sh,
		CommentCollector: collector,
		Tools:            reg,
		MaxConcurrency:   1,
		MaxTokensBudget:  budget,
		SkipFilter:       true,
		Template:         tpl,
		MainToolDefs: []llm.ToolDef{
			{Type: "function", Function: llm.FunctionDef{Name: "task_done", Description: "done"}},
			{Type: "function", Function: llm.FunctionDef{Name: "code_comment", Description: "comment"}},
		},
	})
	a.diffs = diffs
	a.currentDate = "2026-11-11 12:00"
	a.args.Tools.Freeze()
	return a
}

// Repro 1: a file whose review was cut off by the token budget after its first
// finding is recorded as completed, and a resume then reuses it instead of
// reviewing it again, so the findings it never got to are never produced.
func TestRepro_BudgetTruncatedFileWithOneCommentIsCompletedAndReused(t *testing.T) {
	diff := model.Diff{OldPath: "config.go", NewPath: "config.go", Diff: "+package config\n", Insertions: 1}
	budget := estimateDiffFileTokens(diff) * 4
	client := &scriptedClient{script: []func() *llm.ChatResponse{
		commentTurn(diff.NewPath, false, budget+1), // one finding, no task_done, budget now exceeded
		emptyTurn(1),
	}}
	a := newTruncationReproAgent(t, []model.Diff{diff}, client, budgetAgentTestTemplate(), budget)

	if _, err := a.dispatchSubtasks(context.Background()); err != nil {
		t.Fatalf("dispatchSubtasks: %v", err)
	}
	if !a.BudgetExceeded() {
		t.Fatal("precondition: the review must have been stopped by the budget")
	}
	manifest := finishManifestFlow(t, a)

	t.Logf("terminal_state=%s completed=%d failed=%d", manifest.TerminalState, len(manifest.Coverage.Completed), len(manifest.Coverage.Failed))
	if len(manifest.Coverage.Completed) != 1 {
		t.Fatalf("expected current behavior: truncated file counted as completed, got %+v", manifest.Coverage)
	}
	if manifest.TerminalState != session.StateComplete {
		t.Fatalf("expected current behavior: terminal_state=complete despite the truncation, got %s", manifest.TerminalState)
	}

	state, err := session.LoadReviewResumeState(a.args.RepoDir, a.session.SessionID)
	if err != nil {
		t.Fatalf("load session: %v", err)
	}
	if _, ok := state.ReusableItem(reviewItemFingerprint(a.reviewMode(), diff)); !ok {
		t.Fatal("expected current behavior: truncated file is reusable on resume")
	}
}

// Repro 2: a file the group fully reviewed in round 1 (the round ended in
// task_done) is recorded as failed(budget) when round 2 is cut off mid-way,
// only because round 1 found nothing in it.
func TestRepro_CleanFileFromCompletedRoundIsFailedWhenRoundTwoIsCut(t *testing.T) {
	diffs := []model.Diff{
		{OldPath: "login.go", NewPath: "login.go", Diff: "+package login\n", Insertions: 1},
		{OldPath: "clean.go", NewPath: "clean.go", Diff: "+package clean\n", Insertions: 1},
	}
	var est int64
	for _, d := range diffs {
		est += estimateDiffFileTokens(d)
	}
	budget := est * 4
	client := &scriptedClient{script: []func() *llm.ChatResponse{
		commentTurn("login.go", true, 10), // round 1: one finding on login.go, then task_done
		emptyTurn(budget + 1),             // round 2: usage blows the budget mid-conversation
		emptyTurn(1),
	}}
	tpl := budgetAgentTestTemplate()
	tpl.MaxReviewRounds = 2
	tpl.GroupingMinFiles = 4
	tpl.GroupingBundleLineThreshold = 200 // two small files share one group
	a := newTruncationReproAgent(t, diffs, client, tpl, budget)

	if _, err := a.dispatchSubtasks(context.Background()); err != nil {
		t.Fatalf("dispatchSubtasks: %v", err)
	}
	if len(a.fileGroups) != 1 || len(a.fileGroups[0].Diffs) != 2 {
		t.Fatalf("precondition: both files must share one group, got %+v", a.fileGroups)
	}
	manifest := finishManifestFlow(t, a)

	t.Logf("terminal_state=%s completed=%d failed=%d", manifest.TerminalState, len(manifest.Coverage.Completed), len(manifest.Coverage.Failed))
	if len(manifest.Coverage.Failed) != 1 || manifest.Coverage.Failed[0].Path != "clean.go" {
		t.Fatalf("expected current behavior: clean.go failed, got %+v", manifest.Coverage)
	}
	if got := manifest.Coverage.Failed[0].Classification; got != session.FailureBudget {
		t.Fatalf("classification = %q, want budget", got)
	}
}
