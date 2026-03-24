from kevlar.utils.tokenizer import extract_thinking, strip_thinking


class TestExtractThinking:
    def test_full_block(self):
        text = "<think>step by step reasoning</think>The answer is 42."
        thinking, remaining = extract_thinking(text)
        assert thinking == "step by step reasoning"
        assert remaining == "The answer is 42."

    def test_tail_block(self):
        """Output starts inside think block (prompt ended with <think>)."""
        text = "reasoning here</think>The answer."
        thinking, remaining = extract_thinking(text)
        assert thinking == "reasoning here"
        assert remaining == "The answer."

    def test_unclosed_block(self):
        """Generation ended mid-thinking."""
        text = "<think>partial reasoning that never finished"
        thinking, remaining = extract_thinking(text)
        assert thinking == "partial reasoning that never finished"
        assert remaining == ""

    def test_no_thinking(self):
        text = "Just a normal response."
        thinking, remaining = extract_thinking(text)
        assert thinking == ""
        assert remaining == "Just a normal response."

    def test_whitespace_handling(self):
        text = "<think>\n  some thoughts\n</think>\n\nClean answer."
        thinking, remaining = extract_thinking(text)
        assert thinking == "some thoughts"
        assert remaining == "Clean answer."

    def test_tail_with_whitespace(self):
        text = "  thinking content  </think>  answer  "
        thinking, remaining = extract_thinking(text)
        assert thinking == "thinking content"
        assert remaining == "answer"


class TestStripThinkingUnchanged:
    def test_full_block(self):
        assert strip_thinking("<think>foo</think>bar") == "bar"

    def test_tail_block(self):
        assert strip_thinking("foo</think>bar") == "bar"

    def test_no_thinking(self):
        assert strip_thinking("plain text") == "plain text"
