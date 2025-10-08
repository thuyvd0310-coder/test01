import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf # <--- ĐÃ SỬA: Import thư viện tài chính

# --- 1. Cấu hình Trang và Tiêu đề ---
st.set_page_config(
    page_title="Hệ thống Thẩm định Phương án Vay Vốn",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🥖 Hệ thống Thẩm định Dự án Sản xuất Bánh mì")
st.subheader("Phân tích Hiệu quả Tài chính và Khả năng Trả nợ")

# --- 2. Hàm Tính Toán Tài chính Cốt lõi ---

def calculate_financial_metrics(i0, r, c, tax_rate, wacc, n, loan_ratio, loan_interest):
    """Tính toán NPV, IRR, DSCR và bảng dòng tiền."""
    
    # Giả định: Khấu hao theo phương pháp đường thẳng trong 10 năm
    depreciation = i0 / n
    
    # Tính Vốn vay và Vốn chủ sở hữu
    loan_amount = i0 * loan_ratio
    equity_amount = i0 * (1 - loan_ratio)
    
    # Giả định Trả gốc đều hàng năm
    # Lưu ý: Không trả gốc năm 0 (năm đầu tư)
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
        # DSCR = EBITDA / (Lãi vay + Trả gốc)
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
    # SỬ DỤNG npf.npv và npf.irr để khắc phục lỗi Attribute Error
    npv = npf.npv(wacc, np.array(fcf_values))
    try:
        irr = npf.irr(np.array(fcf_values))
    except:
        irr = np.nan
    
    # DSCR trung bình (chỉ tính từ năm 1)
    avg_dscr = pd.DataFrame(cash_flows[1:])['DSCR'].mean()
    
    return pd.DataFrame(cash_flows), npv, irr, avg_dscr


# --- 3. Sidebar: Khu vực Nhập liệu Đầu vào ---
with st.sidebar:
    st.header("⚙️ Thông số Dự án")

    # Thông tin cơ bản
    I0 = st.number_input("Tổng Vốn Đầu tư (VNĐ)", min_value=1000000000.0, value=30000000000.0, step=1000000000.0, format="%0.0f")
    N = st.slider("Vòng đời Dự án (Năm)", 5, 20, 10)
    
    st.markdown("---")
    
    # Thông số Tài chính
    R = st.number_input("Doanh thu Hàng năm (VNĐ)", min_value=100000000.0, value=3500000000.0, step=100000000.0, format="%0.0f")
    C = st.number_input("Chi phí Hoạt động Hàng năm (VNĐ)", min_value=100000000.0, value=2000000000.0, step=100000000.0, format="%0.0f")
    Tax_Rate = st.slider("Thuế suất TNDN (%)", 0.0, 30.0, 20.0) / 100
    WACC = st.slider("WACC/Chi phí Vốn (%)", 5.0, 25.0, 13.0) / 100
    
    st.markdown("---")
    
    # Thông số Vay vốn
    Loan_Ratio = st.slider("Tỷ lệ Vay Ngân hàng (%)", 0.0, 100.0, 80.0) / 100
    Loan_Interest = st.slider("Lãi suất Vay Ngân hàng (%)", 5.0, 15.0, 10.0) / 100
    
    st.info(f"Khoản Vay Dự kiến: **{Loan_Ratio * I0:,.0f}** VNĐ")
    st.info(f"Tài sản Đảm bảo: **70.000.000.000** VNĐ")

# --- 4. Tính toán và Hiển thị Kết quả ---

# Thực hiện tính toán
df_cashflow, npv_result, irr_result, avg_dscr_result = calculate_financial_metrics(
    I0, R, C, Tax_Rate, WACC, N, Loan_Ratio, Loan_Interest
)

# Hàm định dạng tiền tệ
def format_currency(value):
    return f"{value:,.0f}"

# --- 4.1. Hiển thị Chỉ số Hiệu quả Cốt lõi ---
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

# --- 4.2. Bảng Dòng tiền Chi tiết ---
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

# --- 4.3. Trực quan hóa Dòng tiền Tích lũy ---
st.header("📈 Đồ thị Dòng tiền Tích lũy")
st.line_chart(df_cashflow.set_index('Năm')['FCF Tích lũy'])

st.markdown("---")

# --- 5. Phân tích Độ nhạy (Sensitivity Analysis) ---
st.header("🔬 Phân tích Độ nhạy (Kịch bản ±10%)")
st.caption("Kiểm tra sự thay đổi của NPV khi Doanh thu và Chi phí thay đổi")

# Tính toán các kịch bản
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
df_sensitivity['NPV (VNĐ)'] = df_sensitivity['NPV (VNĐ)'].apply(lambda x: f"{x:,.0f}")
df_sensitivity['IRR (%)'] = df_sensitivity['IRR (%)'].apply(lambda x: f"{x:.2f}")
df_sensitivity['DSCR TB'] = df_sensitivity['DSCR TB'].apply(lambda x: f"{x:.2f}")
df_sensitivity['Doanh thu (R)'] = df_sensitivity['Doanh thu (R)'].apply(lambda x: f"{x:,.0f}")
df_sensitivity['Chi phí (C)'] = df_sensitivity['Chi phí (C)'].apply(lambda x: f"{x:,.0f}")

st.table(df_sensitivity)

st.markdown("---")

# --- 6. Khu vực AI Insight (Placeholder) ---
st.header("🧠 Nhận định Chuyên môn (AI Insight)")
st.info("""
    **[CHỖ DÀNH CHO TÍCH HỢP AI]**
    Để kích hoạt tính năng này, bạn cần sử dụng API của mô hình AI (như Google Gemini hoặc OpenAI) và đưa các kết quả tài chính (NPV, IRR, DSCR) vào prompt.
    
    Ví dụ:
    **Prompt:** "Phân tích dự án có NPV {npv_result}, IRR {irr_result} so với WACC {WACC}. Đánh giá rủi ro dựa trên kịch bản bi quan."
""")
