import os
import json
from datetime import datetime


class EpisodicMemory:
    """
    Stores and retrieves summaries of past conversations.
    
    Each session gets summarized at the end and saved to a JSON file.
    At the start of a new session, past summaries are loaded and
    injected into the conversation as context — so the agent
    "remembers" previous interactions.
    
    Storage: simple JSON file (no database needed for learning).
    """

    def __init__(self, storage_path: str = "memory_store.json"):
        self.storage_path = storage_path
        self.memories: list[dict] = []
        self._load()

    def _load(self):
        """Load existing memories from disk."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                self.memories = json.load(f)
        else:
            self.memories = []

    def _save(self):
        """Persist memories to disk."""
        with open(self.storage_path, "w") as f:
            json.dump(self.memories, f, indent=2)

    def save_session(self, summary: str):
        """
        Save a summary of the current session.
        Called at the end of each conversation.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }
        self.memories.append(entry)
        self._save()
        print(f"[Memory] Session saved: {summary[:60]}...")

    def build_context(self, max_sessions: int = 3) -> str:
        """
        Build a context string from the last N sessions.
        This gets injected at the start of each new conversation.
        
        max_sessions: how many past sessions to include
                      (too many = wastes context window)
        """
        if not self.memories:
            return ""

        # Take the most recent N sessions
        recent = self.memories[-max_sessions:]

        context_lines = ["[Past conversation context]"]
        for mem in recent:
            ts = mem["timestamp"][:10]  # just the date
            context_lines.append(f"- {ts}: {mem['summary']}")

        return "\n".join(context_lines)

    def has_memories(self) -> bool:
        return len(self.memories) > 0

    def clear(self):
        """Wipe all memories — useful for testing."""
        self.memories = []
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        print("[Memory] All memories cleared.")