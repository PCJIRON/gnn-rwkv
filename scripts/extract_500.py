import os

with open("huge_hinglish_corpus.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

formatted_chats = []
count = 0
for i in range(0, len(lines) - 1, 2):
    if count >= 500:
        break
    user_line = lines[i]
    brain_line = lines[i+1]
    
    # Simple cleaning
    if len(user_line) > 5 and len(brain_line) > 5:
        formatted_chats.append(f"User: {user_line}")
        formatted_chats.append(f"Brain: {brain_line}")
        count += 1

with open("clean_chat_500.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(formatted_chats))

print(f"Extracted {count} high-quality chat pairs into clean_chat_500.txt")
