import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
from google.genai.client import Client # <--- IMPORT THƯ VIỆN GENAI

# --- 0. Hàm Tích hợp AI (NEW) ---

@st.cache_data
def get_ai_insight(npv, irr, wacc, dscr, sensitivity_results):
    """Gọi Gemini API để phân tích kết quả tài chính."""
    
    # Kiểm tra khóa API (Được lưu trong Streamlit Secrets)
    if "GEMINI_API_KEY" not in st.secrets:
        return (
            "⚠️ **Lỗi Cấu hình AI:** Vui lòng thiết lập khóa `GEMINI_API_KEY` "
            "trong Streamlit Secrets để kích hoạt tính năng phân tích chuyên môn."
        )

    try:
        # Khởi tạo Client
        client = Client(api_key=st.secrets["GEMINI_API_KEY"])

        # Chuẩn bị Prompt
        prompt = f"""
        Bạn là một chuyên gia thẩm định dự án tài chính cấp cao. Nhiệm vụ của bạn là phân tích và đưa ra nhận định chuyên môn ngắn gọn (dưới 150 từ) về tính khả thi và rủi ro của dự án đầu tư dây chuyền sản xuất bánh mì dựa trên các chỉ số sau.

        Các Chỉ số Chính:
        - NPV: {npv:,.0f} VNĐ
        - IRR: {irr*100:.2f}%
        - WACC (Chi phí vốn): {wacc*100:.2f}%
        - DSCR (Khả năng trả nợ) Trung bình: {dscr:.2f}

        Phân tích Độ nhạy (Kết quả NPV trong các kịch bản):
        {sensitivity_results.to_markdown(index=False)}

        Yêu cầu Phân tích:
        1.  Dự án có khả thi không? (So sánh IRR với WACC và NPV > 0).
        2.  Mức độ an toàn của khả năng trả nợ (DSCR).
        3.  Đánh giá rủi ro dựa trên kịch bản Bi quan.
        """

        # Gọi API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"❌ **Lỗi gọi API Gemini:** {e}"


# --- 1. Cấu hình Trang và Tiêu đề ---
st.set_page_config(
    page_title="Hệ thống Thẩm định Phương án Vay Vốn",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🥖 Hệ thống Thẩm định Dự án Sản xuất Bánh mì")
st.subheader("Phân tích Hiệu quả Tài chính và Khả năng Trả nợ")

# --- 2. Hàm Tính Toán Tài chính Cốt lõi (Giữ nguyên) ---

def calculate_financial_metrics(i0, r, c, tax_rate, wacc, n, loan_ratio, loan_interest):
    """Tính toán NPV, IRR, DSCR và bảng dòng tiền."""
    
    # Giả định: Khấu hao theo phương pháp đường thẳng trong 10 năm
    depreciation = i0 / n
    
    # Tính Vốn vay và Vốn chủ sở hữu
    loan_amount = i0 * loan_ratio
    equity_amount = i0 * (1 - loan_ratio)
    
    # Giả định Trả gốc đều hàng năm
    principal_repayment = loan_amount / n
    
    cash_flows = []
    
    # Dòng tiền năm 0 (Đầu tư)
    cash_flows.append({
        'Năm': 0,
        'Doanh thu (R)': 0,
        'Chi phí (C)': 0,
        'EBITDA': 0,
        'Khấu hao (D)': 0,
        'Lãi vay (I)': 0,
        'EBIT': 0,
        'Thuế (20%)': 0,
        'Lợi nhuận ròng': 0,
        'Trả gốc': 0,
        'FCF (Dòng tiền tự do)': -i0,
        'FCF Tích lũy': -i0,
        'DSCR': 0
    })

    # Dòng tiền từ năm 1 đến năm N
    fcf_values = [-i0]
    cumulative_fcf = -i0

    for year in range(1, n + 1):
        # Lãi vay tính trên dư nợ gốc
        outstanding_principal = loan_amount - principal_repayment * (year - 1) if year <= n else 0
        interest_expense = outstanding_principal * loan_interest
        
        # 1. Các chỉ số cơ bản
        ebitda = r - c
        ebit = ebitda - depreciation - interest_expense
        
        # 2. Thuế và Lợi nhuận
        tax = ebit * tax_rate if ebit > 0 else 0
        net_income = ebit - tax
        
        # 3. Dòng tiền Tự do (Dòng tiền cho Chủ sở hữu: Net Income + Khấu hao)
        fcf_for_npv_irr = net_income + depreciation 
        
        cumulative_fcf += fcf_for_npv_irr
        fcf_values.append(fcf_for_npv_irr)

        # 4. Tính toán DSCR
        debt_service = interest_expense + principal_repayment
        dscr = ebitda / debt_service if debt_service > 0 else float('inf')
        
        cash_flows.append({
            'Năm': year,
            'Doanh thu (R)': r,
            'Chi phí (C)': c,
            'EBITDA': ebitda,
            'Khấu hao (D)': depreciation,
            'Lãi vay (I)': interest_expense,
            'EBIT': ebit,
            'Thuế (20%)': tax,
            'Lợi nhuận ròng': net_income,
            'Trả gốc': principal_repayment,
            'FCF (Dòng tiền tự do)': fcf_for_npv_irr,
            'FCF Tích lũy': cumulative_fcf,
            'DSCR': dscr
        })
    
    # Tính NPV và IRR
    npv = npf.npv(wacc, np.array(fcf_values))
    try:
        irr = npf.irr(np.array(fcf_values))
    except:
        irr = np.nan
    
    # DSCR trung bình (chỉ tính từ năm 1)
    avg_dscr = pd.DataFrame(cash_flows[1:])['DSCR'].mean()
    
    return pd.DataFrame(cash_flows), npv, irr, avg_dscr


# --- 3. Sidebar: Khu vực Nhập liệu Đầu vào (Giữ nguyên) ---
with st.sidebar:
    st.header("⚙️ Thông số Dự án")

    I0 = st.number_input("Tổng Vốn Đầu tư (VNĐ)", min_value=1000000000.0, value=30000000000.0, step=1000000000.0, format="%0.0f")
    N = st.slider("Vòng đời Dự án (Năm)", 5, 20, 10)
    
    st.markdown("---")
    
    R = st.number_input("Doanh thu Hàng năm (VNĐ)", min_value=100000000.0, value=3500000000.0, step=100000000.0, format="%0.0f")
    C = st.number_input("Chi phí Hoạt động Hàng năm (VNĐ)", min_value=100000000.0, value=2000000000.0, step=100000000.0, format="%0.0f")
    Tax_Rate = st.slider("Thuế suất TNDN (%)", 0.0, 30.0, 20.0) / 100
    WACC = st.slider("WACC/Chi phí Vốn (%)", 5.0, 25.0, 13.0) / 100
    
    st.markdown("---")
    
    Loan_Ratio = st.slider("Tỷ lệ Vay Ngân hàng (%)", 0.0, 100.0, 80.0) / 100
    Loan_Interest = st.slider("Lãi suất Vay Ngân hàng (%)", 5.0, 15.0, 10.0) / 100
    
    st.info(f"Khoản Vay Dự kiến: **{Loan_Ratio * I0:,.0f}** VNĐ")
    st.info(f"Tài sản Đảm bảo: **70.000.000.000** VNĐ")

# --- 4. Tính toán và Hiển thị Kết quả (Giữ nguyên) ---
df_cashflow, npv_result, irr_result, avg_dscr_result = calculate_financial_metrics(
    I0, R, C, Tax_Rate, WACC, N, Loan_Ratio, Loan_Interest
)

def format_currency(value):
    return f"{value:,.0f}"

st.header("✨ Các Chỉ số Hiệu quả Tài chính")
col1, col2, col3 = st.columns(3)

col1.metric(
    label="Hiện giá Thuần (NPV)",
    value=f"{npv_result:,.0f} VNĐ",
    delta="Dự án Khả thi" if npv_result > 0 else "Cần xem xét"
)

col2.metric(
    label="Tỷ suất Hoàn vốn Nội bộ (IRR)",
    value=f"{irr_result * 100:.2f} %",
    delta=f"Vượt WACC: {WACC*100:.2f}%" if irr_result > WACC else "Thấp hơn WACC"
)

col3.metric(
    label="Khả năng Trả nợ (DSCR TB)",
    value=f"{avg_dscr_result:.2f}",
    delta="An toàn (>1.25)" if avg_dscr_result >= 1.25 else "Rủi ro"
)

st.markdown("---")

st.header("📊 Bảng Dòng tiền Tự do (FCF) Chi tiết")
st.dataframe(
    df_cashflow.style.format({
        'Doanh thu (R)': format_currency,
        'Chi phí (C)': format_currency,
        'EBITDA': format_currency,
        'Khấu hao (D)': format_currency,
        'Lãi vay (I)': format_currency,
        'EBIT': format_currency,
        'Thuế (20%)': format_currency,
        'Lợi nhuận ròng': format_currency,
        'Trả gốc': format_currency,
        'FCF (Dòng tiền tự do)': format_currency,
        'FCF Tích lũy': format_currency,
        'DSCR': "{:.2f}"
    }),
    use_container_width=True,
    height=450
)

st.markdown("---")

# --- 5. Phân tích Độ nhạy (Sensitivity Analysis - Tạo DataFrame để gửi cho AI) ---
st.header("🔬 Phân tích Độ nhạy (Kịch bản ±10%)")
scenarios = {
    'Lạc quan (+10% R, -10% C)': (R * 1.1, C * 0.9),
    'Cơ sở (Base Case)': (R, C),
    'Bi quan (-10% R, +10% C)': (R * 0.9, C * 1.1)
}

sensitivity_results = []
for name, (r_scen, c_scen) in scenarios.items():
    _, npv_scen, irr_scen, dscr_scen = calculate_financial_metrics(
        I0, r_scen, c_scen, Tax_Rate, WACC, N, Loan_Ratio, Loan_Interest
    )
    sensitivity_results.append({
        'Kịch bản': name,
        'Doanh thu (R)': r_scen,
        'Chi phí (C)': c_scen,
        'NPV (VNĐ)': npv_scen,
        'IRR (%)': irr_scen * 100,
        'DSCR TB': dscr_scen
    })

df_sensitivity = pd.DataFrame(sensitivity_results)
df_display_sensitivity = df_sensitivity.copy() # Dùng DF này để hiển thị

df_display_sensitivity['NPV (VNĐ)'] = df_display_sensitivity['NPV (VNĐ)'].apply(lambda x: f"{x:,.0f}")
df_display_sensitivity['IRR (%)'] = df_display_sensitivity['IRR (%)'].apply(lambda x: f"{x:.2f}")
df_display_sensitivity['DSCR TB'] = df_display_sensitivity['DSCR TB'].apply(lambda x: f"{x:.2f}")
df_display_sensitivity['Doanh thu (R)'] = df_display_sensitivity['Doanh thu (R)'].apply(lambda x: f"{x:,.0f}")
df_display_sensitivity['Chi phí (C)'] = df_display_sensitivity['Chi phí (C)'].apply(lambda x: f"{x:,.0f}")

st.table(df_display_sensitivity)
st.markdown("---")

# --- 6. Khu vực AI Insight (ĐÃ TÍCH HỢP GEMINI) ---
st.header("🧠 Nhận định Chuyên môn (AI Insight)")
with st.spinner("Gemini đang phân tích dữ liệu tài chính..."):
    # Gọi hàm phân tích AI
    ai_analysis = get_ai_insight(
        npv_result, 
        irr_result, 
        WACC, 
        avg_dscr_result, 
        df_sensitivity[['Kịch bản', 'NPV (VNĐ)', 'IRR (%)']] # Chỉ gửi các cột quan trọng
    )
    st.markdown(ai_analysis)

# --- 7. Đồ thị (Được chuyển xuống dưới cùng để không làm gián đoạn luồng) ---
st.header("📈 Đồ thị Dòng tiền Tích lũy")
st.line_chart(df_cashflow.set_index('Năm')['FCF Tích lũy'])
