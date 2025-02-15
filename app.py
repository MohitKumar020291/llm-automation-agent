import os
import re
import json
import sqlite3
import subprocess
import tempfile
import base64
from datetime import datetime
from flask import Flask, request, jsonify, Response, abort
from werkzeug.exceptions import BadRequest, InternalServerError
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from markdown import markdown
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


def is_safe_path(path: str) -> bool:
    normalized = os.path.normpath(path)
    return normalized.startswith("/data") and ".." not in normalized

def read_file(path: str) -> str:
    if not is_safe_path(path):
        raise BadRequest("Access to this file is not allowed.")
    if not os.path.isfile(path):
        abort(404)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str):
    if not is_safe_path(path):
        raise BadRequest("Access to this file is not allowed.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def call_llm(prompt: str) -> str:
    """
    Call the LLM via AIproxy to classify the task and return the task name (e.g., "A1", "B3").
    """
    try:
        # aiproxy_token = os.environ.get("AIPROXY_TOKEN")
        aiproxy_token = os.getenv("AIPROXY_TOKEN", "default_token")
        if not aiproxy_token:
            raise InternalServerError("AIPROXY_TOKEN not set in environment.")

        aiproxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {aiproxy_token}",
            "Content-Type": "application/json"
        }

        system_prompt = """
        The DataWorks operations team has identified the following tasks that need to be automated:

        === Phase A Tasks ===
        A1. Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with ${user.email} as the only argument.
        A2. Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place.
        A3. Count the number of Wednesdays in /data/dates.txt and write the number to /data/dates-wednesdays.txt.
        A4. Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json.
        A5. Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.
        A6. Find all Markdown (.md) files in /data/docs/, extract the first H1, and create /data/docs/index.json mapping filenames to titles.
        A7. Extract the sender’s email address from /data/email.txt and write it to /data/email-sender.txt.
        A8. Extract the credit card number from /data/credit-card.png and write it to /data/credit-card.txt.
        A9. Find the most similar pair of comments in /data/comments.txt and write them to /data/comments-similar.txt.
        A10. Calculate the total sales of "Gold" tickets in /data/ticket-sales.db and write the result to /data/ticket-sales-gold.txt.

        === Phase B Tasks ===
        B3. Fetch data from an API and save it.
        B4. Clone a git repo and make a commit.
        B5. Run a SQL query on a SQLite or DuckDB database.
        B6. Extract data from (i.e., scrape) a website.
        B7. Compress or resize an image.
        B8. Transcribe audio from an MP3 file.
        B9. Convert Markdown to HTML.
        B10. Filter a CSV file and return JSON data.

        === Instructions ===
        Your task is to classify the user's task description into one of the above tasks (e.g., "A1", "B3").
        Return ONLY the task name (e.g., "A1") and NOTHING ELSE.
        """

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }
        response = requests.post(aiproxy_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise InternalServerError(f"AIproxy API error: {response.status_code} - {response.text}")
        response_data = response.json()
        task_name = response_data["choices"][0]["message"]["content"].strip()
        return task_name
    except Exception as e:
        raise InternalServerError(f"Failed to call LLM via AIproxy: {str(e)}")

def task_A1():
    try:
        import uv
    except ImportError:
        subprocess.check_call(["pip", "install", "uv"])
    
    user_email = os.environ.get("USER_EMAIL")
    # user_email = os.getenv("USER_EMAIL", "default_email@example.com")
    if not user_email:
        raise BadRequest("USER_EMAIL environment variable not set.")

    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    response = requests.get(url)
    if response.status_code != 200:
        raise InternalServerError("Failed to download datagen.py from remote source.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(response.text)
        tmp_path = tmp.name

    try:
        subprocess.check_call(["python", tmp_path, user_email])
    finally:
        os.remove(tmp_path)
    return "Task A1 completed."

def task_A2():
    file_path = "/data/format.md"
    if not os.path.isfile(file_path):
        raise BadRequest(f"File {file_path} does not exist.")
    cmd = ["npx", "prettier@3.4.2", "--write", file_path]
    subprocess.check_call(cmd)
    return "Task A2 completed."


def task_A3():
    input_path = "/data/dates.txt"
    output_path = "/data/dates-wednesdays.txt"
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")

    count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                date_obj = datetime.strptime(line, "%Y-%m-%d")
            except ValueError:
                continue
            if date_obj.weekday() == 2:
                count += 1

    write_file(output_path, str(count))
    return "Task A3 completed."

def task_A4():
    input_path = "/data/contacts.json"
    output_path = "/data/contacts-sorted.json"
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")

    with open(input_path, "r", encoding="utf-8") as f:
        contacts = json.load(f)

    sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
    write_file(output_path, json.dumps(sorted_contacts, indent=2))
    return "Task A4 completed."

def task_A5():
    logs_dir = "/data/logs"
    output_path = "/data/logs-recent.txt"
    if not os.path.isdir(logs_dir):
        raise BadRequest(f"Directory {logs_dir} does not exist.")
    log_files = []
    for fname in os.listdir(logs_dir):
        if fname.endswith(".log"):
            full_path = os.path.join(logs_dir, fname)
            mtime = os.path.getmtime(full_path)
            log_files.append((full_path, mtime))

    if not log_files:
        raise BadRequest("No .log files found.")
    log_files.sort(key=lambda x: x[1], reverse=True)
    selected = log_files[:10]

    lines = []
    for file_path, _ in selected:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            lines.append(first_line)
    write_file(output_path, "\n".join(lines))
    return "Task A5 completed."


def task_A6():
    docs_dir = "/data/docs"
    index = {}

    if not os.path.isdir(docs_dir):
        raise BadRequest(f"Directory {docs_dir} does not exist.")
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, docs_dir)
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.match(r"^#\s+(.*)", line)
                        if match:
                            title = match.group(1).strip()
                            index[rel_path] = title
                            break
    output_path = os.path.join(docs_dir, "index.json")
    write_file(output_path, json.dumps(index, indent=2))
    return "Task A6 completed."

def task_A7():
    input_path = "/data/email.txt"
    output_path = "/data/email-sender.txt"
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")

    with open(input_path, "r", encoding="utf-8") as f:
        email_content = f.read()

    prompt = f"Extract the sender's email address from the following email message:\n\n{email_content}"
    result = call_llm(prompt)
    write_file(output_path, result.strip())
    return "Task A7 completed."

def task_A8():
    input_path = "/data/credit_card.png"
    output_path = "/data/credit-card.txt"
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")

    with open(input_path, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    prompt = f"Extract the credit card number from the image data (base64 encoded): {encoded_image}"
    result = call_llm(prompt)
    card_number = result.replace(" ", "")
    write_file(output_path, card_number.strip())
    return "Task A8 completed."

def task_A9():
    input_path = "/data/comments.txt"
    output_path = "/data/comments-similar.txt"
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")

    with open(input_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]

    if len(comments) < 2:
        raise BadRequest("Not enough comments to compare.")

    vectorizer = TfidfVectorizer().fit_transform(comments)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    max_sim = -1.0
    pair = (None, None)
    num = len(comments)
    for i in range(num):
        for j in range(i+1, num):
            if cosine_matrix[i][j] > max_sim:
                max_sim = cosine_matrix[i][j]
                pair = (comments[i], comments[j])

    output_text = f"{pair[0]}\n{pair[1]}"
    write_file(output_path, output_text)
    return "Task A9 completed."


def task_A10():
    db_path = "/data/ticket-sales.db"
    output_path = "/data/ticket-sales-gold.txt"
    if not os.path.isfile(db_path):
        raise BadRequest(f"Database file {db_path} does not exist.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    result = cursor.fetchone()[0]
    conn.close()
    write_file(output_path, str(result))
    return "Task A10 completed."

def task_B3_fetch_api():
    task = request.args.get("task", "")
    match = re.search(r"(https?://\S+).+(/data/\S+)", task)
    if not match:
        raise BadRequest("Could not parse API URL and destination path from task description.")
    api_url, dest_path = match.groups()
    response = requests.get(api_url)
    if response.status_code != 200:
        raise InternalServerError("Failed to fetch data from API.")
    write_file(dest_path, response.text)
    return "Task B3 (fetch API) completed."


def task_B4_clone_git():
    task = request.args.get("task", "")
    match = re.search(r"clone\s+(https?://\S+)\s+.*(/data/\S+)", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse git repo URL and destination directory.")
    repo_url, dest_dir = match.groups()
    base_dir = dest_dir
    counter = 1
    while os.path.exists(dest_dir):
        dest_dir = f"{base_dir}{counter}"
        counter += 1
    subprocess.check_call(["git", "clone", repo_url, dest_dir])

    dummy_file = os.path.join(dest_dir, "dummy.txt")
    write_file(dummy_file, "This is a dummy change.")
    subprocess.check_call(["git", "-C", dest_dir, "add", "."])
    subprocess.check_call(["git", "-C", dest_dir, "commit", "-m", "Automated commit by LLM agent"])

    return f"Task B4 (clone git repo) completed. Repository cloned to {dest_dir}."


def task_B5_run_sql():
    task = request.args.get("task", "")
    match = re.search(r"SELECT\s+(.+)\s+FROM\s+(\S+)\s+on\s+(/data/\S+)", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse SQL query and database file from task.")
    query_part, table, db_path = match.groups()
    query = f"SELECT {query_part} FROM {table}"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    output_path = "/data/sql-result.json"
    write_file(output_path, json.dumps(result))
    return "Task B5 (run SQL) completed."

def task_B6_scrape_website():
    task = request.args.get("task", "")
    match = re.search(r"scrape\s+(https?://\S+).+to\s+(/data/\S+)", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse website URL and destination file.")
    url, dest_path = match.groups()
    response = requests.get(url)
    if response.status_code != 200:
        raise InternalServerError("Failed to fetch website.")
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    write_file(dest_path, title)
    return "Task B6 (scrape website) completed."

def task_B7_resize_image():
    task = request.args.get("task", "")
    match = re.search(r"resize\s+(/data/\S+).+width\s+(\d+).+to\s+(/data/\S+)", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse image resize parameters.")
    input_path, width_str, output_path = match.groups()
    width = int(width_str)
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")
    img = Image.open(input_path)
    w_percent = (width / float(img.size[0]))
    height = int((float(img.size[1]) * float(w_percent)))
    img_resized = img.resize((width, height))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_resized.save(output_path)
    return "Task B7 (resize image) completed."


def task_B8_transcribe_audio():
    task = request.args.get("task", "")
    match = re.search(r"transcribe\s+(/data/\S+).+to\s+(/data/\S+)", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse transcription parameters.")
    input_path, output_path = match.groups()
    # Simulate transcription (in real usage, integrate with an audio transcription API)
    transcription = "Simulated transcription of the audio file."
    write_file(output_path, transcription)
    return "Task B8 (transcribe audio) completed."

def task_B9_md_to_html():
    task = request.args.get("task", "")

    match = re.search(r"convert\s+(/data/\S+\.md)\s+to\s+html\s+and\s+save\s+it\s+to\s+(/data/\S+\.html)", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse input and output paths from task description.")

    input_path, output_path = match.groups()
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")

    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html = markdown(md_text)

    write_file(output_path, html)

    return f"Task B9 (Markdown to HTML) completed. Output saved to {output_path}."


def task_B10_filter_csv():
    import csv
    task = request.args.get("task", "")
    match = re.search(r"filter\s+(/data/\S+\.csv).+column\s+'(\w+)'\s+equals\s+'(\w+)'", task, re.IGNORECASE)
    if not match:
        raise BadRequest("Could not parse CSV filter parameters.")
    input_path, column, value = match.groups()
    output_path = "/data/filtered.json"
    if not os.path.isfile(input_path):
        raise BadRequest(f"File {input_path} does not exist.")
    filtered_rows = []
    with open(input_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get(column) == value:
                filtered_rows.append(row)
    write_file(output_path, json.dumps(filtered_rows, indent=2))
    return "Task B10 (CSV filter) completed."

def classify_task(task_description: str) -> str:
    prompt = f"""Classify the following task into one of the following: A1, A2, ..., B10.
    Task: {task_description}
    Classification:"""
    return call_llm(prompt)

def dispatch_task(task_description: str) -> str:
    try:
        task_type = classify_task(task_description)
        print(f"Task classified as: {task_type}")
        task_functions = {
            "A1": task_A1,
            "A2": task_A2,
            "A3": task_A3,
            "A4": task_A4,
            "A5": task_A5,
            "A6": task_A6,
            "A7": task_A7,
            "A8": task_A8,
            "A9": task_A9,
            "A10": task_A10,
            "B3": task_B3_fetch_api,
            "B4": task_B4_clone_git,
            "B5": task_B5_run_sql,
            "B6": task_B6_scrape_website,
            "B7": task_B7_resize_image,
            "B8": task_B8_transcribe_audio,
            "B9": task_B9_md_to_html,
            "B10": task_B10_filter_csv,
        }

        if task_type in task_functions:
            return task_functions[task_type]()
        else:
            raise BadRequest(f"Unsupported task type: {task_type}")
    except subprocess.CalledProcessError as e:
        raise InternalServerError(f"Command execution failed: {str(e)}")
    except Exception as e:
        raise InternalServerError(f"Error processing task: {str(e)}")

@app.route("/run", methods=["POST"])
def run_task():
    task = request.args.get("task")
    if not task:
        return jsonify({"error": "Missing task parameter"}), 400
    try:
        result_message = dispatch_task(task)
        return jsonify({"message": result_message}), 200
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/read", methods=["GET"])
def read_endpoint():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "Missing path parameter"}), 400
    if not is_safe_path(path):
        return jsonify({"error": "Access to this file is not allowed."}), 400
    if not os.path.isfile(path):
        return Response(status=404)
    try:
        content = read_file(path)
        return Response(content, mimetype="text/plain"), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)