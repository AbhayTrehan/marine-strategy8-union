class _FakeConv:
    def __init__(self):
        self.roles = ("USER", "ASSISTANT")
        self._messages = []

    def copy(self):
        return _FakeConv()

    def append_message(self, role, value):
        self._messages.append((role, value))

    def get_prompt(self):
        parts = []
        for role, value in self._messages:
            if value is None:
                parts.append(f"{role}:")
            else:
                parts.append(f"{role}: {value}")
        return " ".join(parts)


conv_templates = {"vicuna_v1": _FakeConv()}
