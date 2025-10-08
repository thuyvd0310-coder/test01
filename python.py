import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from docx import Document
from google import genai
from google.genai.errors import APIError
import json
import math

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Dự Án Kinh Doanh (AI-Powered)",
    layout="wide"
)

st.title("Ứng Dụng Đánh Giá Phương Án Kinh Doanh (AI-Powered) 🚀")

# Khởi tạo state để lưu trữ dữ liệu đã lọc
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

# --- Khai báo các hàm Phân tích Tài chính ---

def calculate_npv(rate, cash_flows):
    """Tính Giá trị Hiện tại Thuần (NPV)"""
    return np.npv(rate, cash_flows)

def calculate_irr(cash_flows):
    """Tính Tỷ suất Hoàn vốn Nội bộ (IRR)"""
    # np.irr yêu cầu cash_flows phải là list/array
    try:
        return np.irr(cash_flows)
    except:
        return np.nan

def calculate_payback_period(cash_flows, discounted=False, rate=None):
    """Tính Thời gian Hoàn vốn (PP) hoặc Thời gian Hoàn vốn có Chiết khấu (DPP)"""
    if discounted and rate is None:
        return None # Cần WACC cho DPP

    cumulative_cf = 0
    periods = 0
    initial_investment = cash_flows[0] # Khoản đầu tư ban đầu (âm)
    
    # Bỏ qua CF ban đầu để tính luỹ kế
    cf_data = cash_flows[1:]
    
    # Nếu tính DPP, chiết khấu dòng tiền
    if discounted and rate is not None:
        discounted_cf = []
        for t, cf in enumerate(cf_data, 1):
            discounted_cf.append(cf / ((1 + rate) ** t))
        cf_data = discounted_cf
    
    # Tính toán thời gian hoàn vốn
    for i, cf in enumerate(cf_data):
        periods += 1
        cumulative_cf += cf
        
        # Kiểm tra xem luỹ kế đã vượt qua vốn đầu tư ban đầu chưa
        if cumulative_cf >= abs(initial_investment):
            # Tính toán phần thập phân (nếu có)
            previous_cumulative = cumulative_cf - cf
            remaining = abs(initial_investment) - previous_cumulative
            fraction = remaining / cf if cf != 0 else 1
            return periods - 1 + fraction
    
    # Nếu không bao giờ hoàn vốn
    return periods + 1

# --- Hàm gọi API Gemini cho Trích xuất Dữ liệu (Task 1) ---

def extract_project_params(doc_text, api_key):
    """Sử dụng Gemini để lọc các chỉ số tài chính từ văn bản."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash-preview-05-20'

        # Định nghĩa Schema cho đầu ra JSON bắt buộc
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "Vốn đầu tư ban đầu (triệu VND)": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu, nhập giá trị dương."},
                "Dòng đời dự án (năm)": {"type": "INTEGER", "description": "Số năm dự án hoạt động, nhập số nguyên."},
                "Doanh thu hàng năm (triệu VND)": {"type": "NUMBER", "description": "Doanh thu hàng năm trung bình."},
                "Chi phí hoạt động hàng năm (triệu VND)": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm trung bình, không bao gồm khấu hao."},
                "WACC (%)": {"type": "NUMBER", "description": "Chi phí sử dụng vốn bình quân, nhập theo phần trăm (ví dụ: 10 cho 10%)."},
                "Thuế suất (%)": {"type": "NUMBER", "description": "Thuế thu nhập doanh nghiệp, nhập theo phần trăm (ví dụ: 20 cho 20%)."}
            },
            "required": ["Vốn đầu tư ban đầu (triệu VND)", "Dòng đời dự án (năm)", "Doanh thu hàng năm (triệu VND)", "Chi phí hoạt động hàng năm (triệu VND)", "WACC (%)", "Thuế suất (%)"]
        }
        
        system_prompt = """
        Bạn là một chuyên gia tài chính, nhiệm vụ của bạn là trích xuất 6 thông số tài chính chính xác sau đây từ văn bản đề xuất kinh doanh được cung cấp: Vốn đầu tư ban đầu, Dòng đời dự án (số năm), Doanh thu hàng năm, Chi phí hoạt động hàng năm, WACC, và Thuế suất. 
        Đầu ra của bạn PHẢI là một đối tượng JSON tuân thủ schema đã định nghĩa.
        Đảm bảo đơn vị là (triệu VND) cho các giá trị tiền tệ và (%) cho WACC và Thuế suất, và (năm) cho dòng đời dự án.
        """
        user_query = f"Trích xuất các thông số tài chính từ văn bản đề xuất kinh doanh sau:\n\n---\n\n{doc_text}"

        response = client.models.generate_content(
            model=model_name,
            contents=user_query,
            system_instruction=system_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )
        
        # Phân tích chuỗi JSON nhận được
        data_json = json.loads(response.text)
        return data_json

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi phân tích JSON từ AI. Vui lòng thử lại hoặc điều chỉnh đề xuất Word file.")
        st.info(f"Phản hồi thô của AI: {response.text}")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None

# --- Hàm gọi API Gemini cho Phân tích (Task 4) ---

def analyze_project_metrics(metrics_data, api_key):
    """Gửi các chỉ số đánh giá dự án đến Gemini để nhận phân tích."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Chuyển đổi dữ liệu metrics thành chuỗi dễ đọc
        metrics_str = "\n".join([f"- {k}: {v}" for k, v in metrics_data.items()])

        prompt = f"""
        Bạn là một chuyên gia tư vấn đầu tư cấp cao. Dựa trên các chỉ số đánh giá hiệu quả dự án sau, hãy đưa ra một đánh giá chi tiết, khách quan và chuyên nghiệp (khoảng 3-4 đoạn) về khả năng chấp nhận đầu tư của dự án. 
        Hãy tập trung vào:
        1. Tính thanh khoản và khả năng tạo giá trị (dựa trên NPV và IRR so với WACC).
        2. Rủi ro và thời gian thu hồi vốn (dựa trên PP và DPP).
        3. Kết luận về việc có nên chấp nhận hay từ chối dự án.

        Các chỉ số dự án:
        {metrics_str}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi yêu cầu phân tích AI: {e}"

# --- Logic Chính của Ứng dụng ---

# 1. Tải File Word
st.markdown("---")
uploaded_file = st.file_uploader(
    "1. Tải file Word (.docx) chứa đề xuất phương án kinh doanh:",
    type=['docx']
)

# Sidebar cho API Key
with st.sidebar:
    st.header("Cấu hình API Key")
    # Sử dụng st.secrets nếu triển khai trên Streamlit Cloud
    st.info("Ứng dụng này cần 'GEMINI_API_KEY' trong Streamlit Secrets để hoạt động.")
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.warning("Vui lòng cấu hình Khóa API trong Secrets hoặc chạy cục bộ.")


if uploaded_file is not None:
    st.success(f"Đã tải file: {uploaded_file.name}")
    
    if st.button("2. Lọc Dữ Liệu Tài Chính Bằng AI (Task 1) 🤖", type="primary"):
        if not api_key:
            st.error("Không tìm thấy GEMINI_API_KEY. Vui lòng kiểm tra cấu hình.")
        else:
            with st.spinner('Đang đọc file và gửi yêu cầu trích xuất dữ liệu đến AI...'):
                
                # Đọc nội dung file Word
                doc_file = BytesIO(uploaded_file.read())
                document = Document(doc_file)
                full_text = []
                for paragraph in document.paragraphs:
                    full_text.append(paragraph.text)
                doc_text = "\n".join(full_text)
                
                # Gọi hàm trích xuất
                extracted_data = extract_project_params(doc_text, api_key)
                st.session_state.extracted_data = extracted_data
    
    # Hiển thị dữ liệu đã lọc và thực hiện tính toán nếu có dữ liệu
    if st.session_state.extracted_data:
        data = st.session_state.extracted_data
        
        st.markdown("---")
        st.subheader("3. Các Thông Số Dự Án Đã Lọc (AI Trích xuất)")
        
        col_params_1, col_params_2, col_params_3 = st.columns(3)
        
        # Hiển thị các thông số quan trọng
        with col_params_1:
            st.metric("Vốn đầu tư ban đầu (triệu VND)", f"{data.get('Vốn đầu tư ban đầu (triệu VND)', 0):,.0f}")
            st.metric("WACC (%)", f"{data.get('WACC (%)', 0):.2f}%")
        with col_params_2:
            st.metric("Dòng đời dự án (năm)", f"{data.get('Dòng đời dự án (năm)', 0)}")
            st.metric("Thuế suất TNDN (%)", f"{data.get('Thuế suất (%)', 0):.2f}%")
        with col_params_3:
            st.metric("Doanh thu hàng năm (triệu VND)", f"{data.get('Doanh thu hàng năm (triệu VND)', 0):,.0f}")
            st.metric("Chi phí HĐ hàng năm (triệu VND)", f"{data.get('Chi phí hoạt động hàng năm (triệu VND)', 0):,.0f}")

        # --- Chuẩn bị và Tính toán Dòng tiền (Tasks 2 & 3) ---
        
        V_dau_tu = data.get('Vốn đầu tư ban đầu (triệu VND)', 0)
        N_nam = data.get('Dòng đời dự án (năm)', 0)
        Doanh_thu = data.get('Doanh thu hàng năm (triệu VND)', 0)
        Chi_phi = data.get('Chi phí hoạt động hàng năm (triệu VND)', 0)
        WACC = data.get('WACC (%)', 0) / 100 # Chuyển % sang thập phân
        Thue_suat = data.get('Thuế suất (%)', 0) / 100 # Chuyển % sang thập phân

        if N_nam > 0 and WACC > 0 and V_dau_tu > 0:
            
            # Tính toán Dòng tiền ròng hàng năm (Net Cash Flow)
            EBIT = Doanh_thu - Chi_phi
            Thue_phai_nop = EBIT * Thue_suat if EBIT > 0 else 0
            Loi_nhuan_sau_thue = EBIT - Thue_phai_nop
            CF_nam = Loi_nhuan_sau_thue # Giả định không có Khấu hao và thay đổi Vốn lưu động

            # Dòng tiền cho tính toán NPV/IRR
            # CFs[0] là Vốn đầu tư ban đầu (âm)
            cash_flows = [-V_dau_tu] + [CF_nam] * N_nam
            
            # Xây dựng Bảng Dòng tiền (Task 2)
            st.subheader("4. Bảng Dòng Tiền Dự Án (Đơn vị: Triệu VND)")
            df_cf = pd.DataFrame({
                'Năm (t)': [0] + list(range(1, N_nam + 1)),
                'Dòng tiền hoạt động ròng (CF)': [0] + [CF_nam] * N_nam,
                'Vốn đầu tư ban đầu': [-V_dau_tu] + [0] * N_nam,
                'Dòng tiền ròng (NCF)': cash_flows
            })
            st.dataframe(df_cf.style.format('{:,.0f}'), use_container_width=True, hide_index=True)
            
            # Tính toán các chỉ số (Task 3)
            npv = calculate_npv(WACC, cash_flows)
            irr = calculate_irr(cash_flows)
            pp = calculate_payback_period(cash_flows)
            dpp = calculate_payback_period(cash_flows, discounted=True, rate=WACC)
            
            st.subheader("5. Các Chỉ Số Đánh Giá Hiệu Quả Dự Án")
            
            col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)
            
            metrics_display = {
                "NPV (Giá trị Hiện tại Thuần)": f"{npv:,.0f} Triệu VND",
                "IRR (Tỷ suất Hoàn vốn Nội bộ)": f"{irr*100:.2f}%" if not math.isnan(irr) else "Không tính được",
                "PP (Thời gian Hoàn vốn)": f"{pp:.2f} năm",
                "DPP (Thời gian Hoàn vốn có Chiết khấu)": f"{dpp:.2f} năm"
            }
            
            with col_metrics_1:
                st.metric("NPV", metrics_display["NPV (Giá trị Hiện tại Thuần)"], 
                          delta="Dự án tạo thêm giá trị" if npv > 0 else "Dự án làm giảm giá trị")
            with col_metrics_2:
                st.metric("IRR", metrics_display["IRR (Tỷ suất Hoàn vốn Nội bộ)"], 
                          delta=f"Cao hơn WACC ({WACC*100:.2f}%)" if (not math.isnan(irr) and irr > WACC) else None)
            with col_metrics_3:
                st.metric("PP", metrics_display["PP (Thời gian Hoàn vốn)"])
            with col_metrics_4:
                st.metric("DPP", metrics_display["DPP (Thời gian Hoàn vốn có Chiết khấu)"])

            # --- Yêu cầu AI Phân tích (Task 4) ---
            st.markdown("---")
            st.subheader("6. Phân Tích Hiệu Quả Dự Án Bằng AI (Task 4)")
            
            # Chuẩn bị dữ liệu cho AI
            metrics_for_ai = {
                "NPV (Triệu VND)": f"{npv:,.2f}",
                "IRR": f"{irr*100:.2f}%" if not math.isnan(irr) else "Không xác định",
                "WACC": f"{WACC*100:.2f}%",
                "PP (năm)": f"{pp:.2f}",
                "DPP (năm)": f"{dpp:.2f}",
                "Dòng đời dự án (năm)": N_nam
            }

            if st.button("Yêu cầu AI Phân tích Hiệu quả Dự án 🧠", key="analyze_button"):
                if api_key:
                    with st.spinner('Đang gửi các chỉ số và chờ Gemini phân tích...'):
                        ai_result = analyze_project_metrics(metrics_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        else:
            st.warning("Không thể tính toán: Dòng đời dự án, WACC, hoặc Vốn đầu tư chưa được trích xuất chính xác.")

elif st.session_state.extracted_data:
    # Xóa dữ liệu cũ nếu người dùng chưa tải file mới
    st.session_state.extracted_data = None
    st.info("Vui lòng tải lên file Word để bắt đầu phân tích.")

else:
    st.info("Vui lòng tải lên file Word (.docx) chứa đề xuất kinh doanh và nhấn nút 'Lọc Dữ Liệu' để bắt đầu.")
