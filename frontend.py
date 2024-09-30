# frontend.py
import tkinter as tk
from queue import Queue, Empty

class ConversationApp:
    def __init__(self, root, command_queue, update_queue):
        self.command_queue = command_queue
        self.update_queue = update_queue
        self.root = root
        self.root.title("Voice Conversation")

        start_button = tk.Button(root, text="Start Conversation", command=self.start_conversation)
        start_button.pack(pady=10)

        stop_button = tk.Button(root, text="Stop Recording", command=self.stop_conversation)
        stop_button.pack(pady=10)

        self.conversation_text = tk.Text(root, height=20, width=50)
        self.conversation_text.pack(pady=10)
        self.conversation_text.insert(tk.END, "Conversation history will appear here...\n")

        quit_button = tk.Button(root, text="Quit", command=root.quit)
        quit_button.pack(pady=10)

        self.root.after(100, self.process_update_queue)

    def start_conversation(self):
        self.command_queue.put("start")

    def stop_conversation(self):
        self.command_queue.put("stop")

    def process_update_queue(self):
        try:
            while True:
                message = self.update_queue.get_nowait()
                self.update_conversation_text(message)
        except Empty:
            pass
        self.root.after(100, self.process_update_queue)

    def update_conversation_text(self, text):
        self.conversation_text.insert(tk.END, text + "\n")
        self.conversation_text.see(tk.END)  # Scroll to the end

def main(command_queue, update_queue):
    root = tk.Tk()
    app = ConversationApp(root, command_queue, update_queue)
    root.mainloop()

if __name__ == "__main__":
    command_queue = Queue()
    update_queue = Queue()
    main(command_queue, update_queue)
