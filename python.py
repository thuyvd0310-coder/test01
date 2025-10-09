# --- Hàm gọi API Gemini ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Chức năng 5: Nhận xét AI ---
st.subheader("5. Nhận xét Tình hình Tài chính (AI)")

# Chuẩn bị dữ liệu để gửi cho AI
data_for_ai = pd.DataFrame({
    'Chỉ tiêu': [
        'Toàn bộ Bảng phân tích (dữ liệu thô)', 
        'Tăng trưởng Tài sản ngắn hạn (%)', 
        'Thanh toán hiện hành (N-1)', 
        'Thanh toán hiện hành (N)'
    ],
    'Giá trị': [
        df_processed.to_markdown(index=False),
        f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
        f"{thanh_toan_hien_hanh_N_1}", 
        f"{thanh_toan_hien_hanh_N}"
    ]
}).to_markdown(index=False) 

if st.button("Yêu cầu AI Phân tích"):
    api_key = st.secrets.get("GEMINI_API_KEY") 
    
    if api_key:
        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
            ai_result = get_ai_analysis(data_for_ai, api_key)
            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
            st.info(ai_result)
    else:
         st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
