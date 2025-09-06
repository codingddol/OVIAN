"""
main.py

Streamlit 기반 난소암 조직 이미지 분석 및 PubMed 요약 UI 메인 실행 파일

- 이미지 업로드 기반 분석 기능
- PubMed 검색 및 AI 요약 기능
- SessionState를 통한 Q&A 히스토리 관리
- Attention Map 및 예측 결과 표시
"""

import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import requests
import openai
import torch
import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()  # .env 파일의 환경변수 불러오기

OVIAN_TOKEN = os.environ.get("OVIAN_TOKEN")
OVIAN_IMAGE_KEY = os.environ.get("OVIAN_IMAGE_KEY")

Image.MAX_IMAGE_PIXELS = None

def get_safe_chat_model():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

# 세션 상태 초기화
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "image_inferred" not in st.session_state:
    st.session_state.image_inferred = False

# LLM RAG 프롬프트
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 난소암 병리 이미지를 해석하는 AI 전문가입니다.

다음은 논문에서 발췌한 내용입니다:
---------------------
{context}
---------------------

이 내용을 바탕으로 아래 질문에 답변하세요.
질문: {question}
"""
)

# PubMed Boolean Query 생성
def generate_pubmed_query_from_question(question: str) -> str:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    prompt_str = f"""
아래 설명은 병리 이미지 AI 분석 결과입니다.
이 설명에 맞는 PubMed 검색 Boolean 쿼리를 생성하세요.

- 핵심 질병명, 조직명, 염색법(H&E 등), 병리 용어 중심 키워드를 포함하세요.
        - Boolean 연산자 (AND, OR)와 괄호를 사용하세요.
        - 키워드는 3~6개 사이로 구성하세요.
        - 예시: (histology OR tissue) AND (H&E OR staining) AND (diagnosis OR cancer)

설명:
{question}

출력 형식: Boolean Query
"""

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 병리 이미지 AI 전문가입니다."},
            {"role": "user", "content": prompt_str}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()


# PubMed 검색
def search_pubmed(question: str, max_results: int = 3):
    Entrez.email = "teamovianai@gmail.com"
    try:
        query = generate_pubmed_query_from_question(question)
        st.info(f"🔎 PubMed Boolean Query: {query}")
    except Exception as e:
        st.warning(f"⚠️ Boolean Query 생성 실패 → 원본 질문으로 검색: {e}")
        query = question

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record.get("IdList", [])
        if not ids:
            st.warning("❌ PubMed에서 논문을 찾을 수 없습니다.")
            return []

        unique_summaries = {}  # 중복 제거를 위해 딕셔너리 사용 (key: pmid, value: Document)
        for pmid in ids:
            # 이미 처리한 pmid이면 건너뜁니다.
            if pmid in unique_summaries:
                continue
            
            try:
                summary = Entrez.esummary(db="pubmed", id=pmid, retmode="xml")
                summary_record = Entrez.read(summary)
                title = summary_record[0].get("Title", "[제목 없음]")
                pubdate = summary_record[0].get("PubDate", "")
                year = pubdate.split()[0] if pubdate else ""
                authors = summary_record[0].get("AuthorList", [])
                author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
                fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
                abstract = fetch.read()

                # Document 객체를 생성하여 딕셔너리에 추가합니다.
                doc = Document(
                    page_content=abstract,
                    metadata={"pmid": pmid, "title": title, "year": year, "authors": author_str, "source": "pubmed"}
                )
                unique_summaries[pmid] = doc

            except Exception as e:
                st.warning(f"⚠️ PMID {pmid} 요약 실패: {e}")
        
        # 딕셔너리의 값들(Document 객체)만 리스트로 변환하여 반환합니다.
        return list(unique_summaries.values())
    
    except Exception as e:
        st.error(f"❌ PubMed 검색 실패: {e}")
        return []

# Streamlit UI 시작
st.set_page_config(page_title="Ovarian Cancer RAG", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🔬 난소암 분석 AI 어시스턴트")

uploaded_file = st.file_uploader("조직 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if st.button("분석 실행") and uploaded_file is not None:
    llm = get_safe_chat_model() 
    embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-base-cased-v1.1")

    image_bytes = uploaded_file.read()

    try:
        with st.spinner("1️⃣ Flask 서버로 이미지 전송 및 AI 추론 중..."):
            files = {'file': (uploaded_file.name, image_bytes)}
            headers = {"Authorization": f"Bearer {OVIAN_TOKEN}"}
            response = requests.post("http://localhost:5001/infer", files=files, headers=headers)

        if response.status_code == 200:
            result = response.json()
            pred_class = result["pred_class"]
            softmax_probs = result["softmax_probs"]
            max_prob = max(softmax_probs)
            attention_base64 = result["attention_map_base64"]
            
            # 보안 상태 배지 (서버가 보내준 security 플래그)
            sec = result.get("security", {})
            enc_ok = bool(sec.get("encrypt"))
            probe_ok = bool(sec.get("decrypt_probe"))
            dec_ok = bool(sec.get("decrypt"))

            st.markdown("### 🔐 보안 상태")
            st.write("✅ 암호화 저장 성공!" if enc_ok else "❌ 암호화 저장 실패/미수행")
            st.write("✅ 복호화 유효성 검사 통과!" if probe_ok else "❌ 복호화 유효성 검사 실패")
            st.write("✅ 복호화(실제 처리 스트림) 성공!" if dec_ok else "❌ 복호화(실제 처리 스트림) 실패")
            st.markdown("---")

            st.session_state.image_inferred = True  # ✅ 이미지 예측 완료 표시

            st.success("✅ Flask 서버 추론 완료!")

            # 이미지 + Attention Map 병렬 표시
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("원본 조직 이미지")
                st.image(Image.open(BytesIO(image_bytes)), use_container_width=True)
            with col2:
                st.subheader("AI Attention Map")
                st.image(BytesIO(base64.b64decode(attention_base64)), use_container_width=True)

            label_dict = {0: 'HGSC', 1: 'LGSC', 2: 'CC', 3: 'EC', 4: 'MC'}
            
            probs_percent = [f"{label_dict[i]}: {int(p * 100)}%" for i, p in enumerate(softmax_probs)]
            st.markdown("---")
            st.subheader("📊 Softmax Probabilities")
            for line in probs_percent:
                st.write(f"- {line}")
            
            st.success(f"✅ AI 예측 클래스: {pred_class} ({label_dict[pred_class]})")

            # Softmax 35% 이상일 때만 PubMed
            if max_prob >= 0.35:
                try:
                    with st.spinner("2️⃣ PubMed 관련 논문 검색 중..."):
                        search_term = label_dict[pred_class] + " ovarian cancer"
                        related_papers = search_pubmed(search_term, max_results=3)
                        
                    
                    # AI 요약 답변 통합 출력
                    st.markdown("---")
                    st.subheader("🧠 AI 요약 답변")

                    # 전체 요약
                    all_text = "\n\n".join([doc.page_content for doc in related_papers])
                    summary_prompt = f"""
                    You are a medical research summarization expert.

                    Please summarize the **overall trends and findings** from the following 3 papers about {label_dict[pred_class]} ovarian cancer.

                    ⚠️ Do NOT summarize each document individually.  
                    Provide a unified 2–3 sentence overview only.

                    --- Documents ---
                    {all_text}
                    """
                    
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a medical paper summarization expert."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        max_tokens=500
                    )
                    overall_summary = response.choices[0].message.content.strip()
                    st.markdown("**전체 요약**")
                    st.markdown(f"- {overall_summary}")

                    # 논문별 요약
                    st.markdown("**논문별 요약**")
                    for idx, doc in enumerate(related_papers, start=1):
                        title = doc.metadata.get("title", "[제목 없음]")

                        single_summary_prompt = f"""
                    Summarize this medical abstract in 3-4 sentences for a clinical researcher.

                    --- Abstract ---
                    {doc.page_content}
                    """
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a medical paper summarization expert."},
                                {"role": "user", "content": single_summary_prompt}
                            ],
                            max_tokens=500
                        )
                        single_summary = response.choices[0].message.content.strip()

                        st.markdown(f"**{idx}. 제목: {title}**")
                        st.markdown(f"- 요약: {single_summary}")

                    # 결론 요약
                    conclusion_prompt = f"""
                    Based on the previous summaries, write a single-sentence final conclusion about
                    {label_dict[pred_class]} ovarian cancer.
                    """
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a medical summarizer."},
                            {"role": "user", "content": conclusion_prompt}
                        ],
                        max_tokens=200
                    )
                    conclusion = response.choices[0].message.content.strip()
                    st.markdown("**결론 요약**")
                    st.markdown(f"- {conclusion}")

                    # 관련 논문 출력
                    st.success("✅ PubMed 검색 완료!")
                    st.markdown("---")
                    st.subheader("📄 PubMed 관련 논문")

                    for doc in related_papers:
                        title = doc.metadata.get("title", "[제목 없음]")
                        year = doc.metadata.get("year", "")
                        authors = doc.metadata.get("authors", "")
                        pmid = doc.metadata.get("pmid", "")
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                        st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

                except Exception as e:
                    st.warning(f"⚠️ PubMed 검색 실패: {e}")
            else:
                st.warning(f"❌ 모델 확신 부족 (최대 Softmax 확률 {int(max_prob*100)}%) → PubMed 검색 생략")
        else:
            st.error(f"❌ Flask 추론 에러: {response.text}")
        # 실패 시에도 서버가 보낸 security 플래그가 있으면 표시
            try:
                err = response.json()
                sec = err.get("security", {})
                if sec:
                    st.markdown("### 🔐 보안 상태(실패 시점)")
                    st.write("✅ 암호화 저장 성공!" if sec.get("encrypt") else "❌ 암호화 저장 실패/미수행")
                    st.write("✅ 복호화 유효성 검사 통과!" if sec.get("decrypt_probe") else "❌ 복호화 유효성 검사 실패")
                    st.write("✅ 복호화(실제 처리 스트림) 성공!" if sec.get("decrypt") else "❌ 복호화(실제 처리 스트림) 실패")
                    st.markdown("---")
            except Exception:
                pass    
                    

    except Exception as e:
        st.error(f"🚨 Flask 서버 연결 실패: {e}")
        st.markdown("### 🔐 보안 상태(연결 실패)")
        st.write("❌ 서버 연결 실패로 보안 단계 정보를 받지 못했습니다.")
        st.markdown("---")

    finally:
        try:
            headers = {"Authorization": f"Bearer {OVIAN_TOKEN}"}
            clear_response = requests.post("http://localhost:5001/clear_uploads", headers=headers)
            if clear_response.status_code == 200:
                st.success("✅ 서버 uploads 폴더 정리 완료!")
            else:
                st.warning(f"⚠️ uploads 폴더 정리 실패: {clear_response.text}")
        except Exception as e:
            st.warning(f"⚠️ Flask 서버 정리 요청 실패: {e}")


# main.py 파일의 일부

# 예측 이후에만 텍스트 질문 Q&A 활성화
if st.session_state.image_inferred:
    st.markdown("---")
    st.subheader("💬 추가 질문 (텍스트 기반 RAG Q&A)")

    # st.form으로 텍스트 입력과 제출 버튼을 그룹화합니다.
    with st.form(key="text_rag_form"):
        question = st.text_input(
            "AI 예측 결과에 대한 추가 질문 입력",
            placeholder="예시: What is the treatment for HGSC?",
            key="text_rag_input"
        )
        # st.button 대신 st.form_submit_button을 사용합니다.
        submit_button = st.form_submit_button(label="텍스트 기반 Q&A 실행")

        if submit_button:  # 버튼이 클릭되거나, form 안에서 Enter를 누르면 True가 됩니다.
            if not question.strip():
                st.warning("질문을 입력해주세요.")
            else:
                # 기존의 RAG 실행 및 결과 출력 로직을 이 안으로 이동합니다.
                with st.spinner("PubMed 검색 및 AI 답변 생성 중..."):
                    llm = get_safe_chat_model()
                    embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-base-cased-v1.1")

                    docs = search_pubmed(question=question)
                    if not docs:
                        st.warning("❌ PubMed에서 질문에 대한 논문이 없습니다.")
                    else:
                        splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
                        chunks = splitter.split_documents(docs)
                        vector_db = FAISS.from_documents(chunks, embeddings)
                        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

                        rag_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            chain_type="stuff",
                            chain_type_kwargs={"prompt": prompt_template},
                            return_source_documents=True
                        )
                        result = rag_chain.invoke({"query": question})

                        # 결과 출력
                        st.markdown("---")
                        st.subheader("🖍️ 요약 답변")

                        summary_prompt = f"""
                        You are a medical research summarization expert.

                        The user asked the following question:
                        "{question}"

                        You are given the content of 3 PubMed papers.  
                        Please answer using the following structure:

                        💬 Question Answer  
                        - Provide a clear 2–3 sentence answer to the user's question, based on the papers.

                        📄 Related Paper Summaries  
                        1. Title: ...  
                        - Summary: ...

                        2. Title: ...  
                        - Summary: ...

                        3. Title: ...  
                        - Summary: ...

                        🔚 Final Conclusion  
                        - In 1 sentence, summarize your answer to the question, based on the above papers.

                        --- Papers ---
                        """
                        for idx, doc in enumerate(result["source_documents"], start=1):
                            summary_prompt += f"\n\n{idx}. {doc.page_content}"

                        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a medical paper summarizer."},
                                {"role": "user", "content": summary_prompt}
                            ],
                            max_tokens=500
                        )
                        structured_summary = response.choices[0].message.content.strip()
                        
                         # 1. "Final Conclusion" 처리 (뒤에 있는 것부터 처리해야 순서가 안 꼬입니다)
                        heading_fc = "Final Conclusion"
                        if heading_fc in structured_summary:
                            # "Final Conclusion" 텍스트를 기준으로 문단을 나눕니다.
                            parts = structured_summary.split(heading_fc, 1)
                            # 제목 앞 부분(parts[0])에서 혹시 있을 이모지를 제거합니다.
                            part_before = parts[0].replace("🔚", "").strip()
                            # 답변 내용(parts[1])에서 혹시 있을 콜론(:)이나 공백을 제거합니다.
                            part_after = parts[1].lstrip(': ')
                            # 다시 조립합니다: (앞 내용) + 줄바꿈 + 이모지 + 제목 + 줄바꿈 + 답변
                            structured_summary = part_before + "\n\n" + "🔚 " + heading_fc + "\n" + part_after

                        # 2. "Question Answer" 처리
                        heading_qa = "Question Answer"
                        if heading_qa in structured_summary:
                            parts = structured_summary.split(heading_qa, 1)
                            part_before = parts[0].replace("💬", "").strip()
                            part_after = parts[1].lstrip(': ')
                            structured_summary = part_before + "💬 " + heading_qa + "\n" + part_after
                            
                        st.markdown(structured_summary)

                        st.markdown("---")
                        st.subheader("📄 관련 논문")
                        for doc in result["source_documents"]:
                            pmid = doc.metadata.get("pmid", "")
                            title = doc.metadata.get("title", "[제목 없음]")
                            year = doc.metadata.get("year", "")
                            authors = doc.metadata.get("authors", "")
                            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                            st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

                        st.session_state.qa_history.insert(0, {
                            "question": question,
                            "answer": structured_summary,
                            "sources": result["source_documents"]
                        })
            
else:
    st.info("👆 먼저 이미지를 업로드하고 AI 예측을 완료하세요. 이후에 질문 가능!")

# Q&A 히스토리
if len(st.session_state.qa_history) > 0:
    st.markdown("## 📚 이전 Q&A 기록")
    for idx, entry in enumerate(st.session_state.qa_history):
        with st.expander(f"Q{len(st.session_state.qa_history) - idx}: {entry['question']}"):
            st.write(entry["answer"])
            for doc in entry["sources"]:
                pmid = doc.metadata.get("pmid")
                title = doc.metadata.get("title", "[제목 없음]")
                year = doc.metadata.get("year", "")
                authors = doc.metadata.get("authors", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

# 초기화 버튼
if st.button("기록 초기화"):
    st.session_state.qa_history = []
    st.session_state.image_inferred = False
    st.rerun()
