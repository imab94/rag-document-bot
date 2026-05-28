def clean_response(answer):
    answer = answer.replace("**", "")
    answer = answer.replace("##", "")
    answer = answer.replace("```", "")
    return answer.strip()

def extract_unique_sources(context):
    sources = set()

    for line in context.splitlines():
        if "SOURCE DOCUMENT" in line:
            source = line.split(":")[-1].strip()
            sources.add(source)
    return sorted(list(sources))

def detect_metadata_filter(question):
    question = question.lower()
    if "pdf" in question:
        return {"document_type": "pdf"}

    if "txt" in question:
        return {"document_type": "txt"}

    if "india" in question:
        return {"file_name": "india.pdf"}

    return None