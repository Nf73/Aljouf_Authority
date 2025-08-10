import streamlit as st
from datetime import datetime
import joblib
import pandas as pd

# إعداد الصفحة
st.set_page_config(
    page_title="تقدير مدة معالجة البلاغات – أمانة الجوف",
    layout="centered",
    initial_sidebar_state="expanded"
)

# تنسيق النصوص
st.markdown(
    """
    <style>
    * {
        direction: rtl;
        text-align: right;
        font-family: 'Arial';
    }
    </style>
    """,
    unsafe_allow_html=True
)

# تحميل الملفات
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
features = joblib.load('features.pkl')

# تحميل البيانات مع التخزين المؤقت
@st.cache_data
def load_raw_data():
    df = pd.read_csv("البلاغات 2024.csv", encoding="utf-8-sig")
    df = df.dropna(subset=['التصنيف الرئيسى', 'التصنيف الفرعي', 'التصنيف التخصصي'])
    return df

df_raw = load_raw_data()

# إنشاء العلاقات بين التصنيفات
main_to_sub = df_raw.groupby('التصنيف الرئيسى')['التصنيف الفرعي'].unique().to_dict()
sub_to_spec = df_raw.groupby('التصنيف الفرعي')['التصنيف التخصصي'].unique().to_dict()

# أسماء البلديات
بلديات = label_encoders['البلدية'].classes_.tolist()

# الشريط الجانبي
st.sidebar.title("حول الخدمة")
st.sidebar.info("""
هذه الخدمة توفر للمواطن المدة التقديرية لمعالجة بلاغ التشوه البصري
باستخدام الذكاء الاصطناعي، بناءً على بيانات تاريخية من أمانة الجوف.
""")

# الشعار
st.image("big_logo.png", width=200)

# العنوان
st.title("تقدير مدة معالجة بلاغ التشوه البصري – أمانة الجوف")

# عرض القوائم جنب بعض
col1, col2 = st.columns(2)
with col1:
    تصنيف_رئيسي = st.selectbox("التصنيف الرئيسي", list(main_to_sub.keys()))
with col2:
    خيارات_الفرعي = main_to_sub.get(تصنيف_رئيسي, [])
    تصنيف_فرعي = st.selectbox("التصنيف الفرعي", خيارات_الفرعي)

col3, col4 = st.columns(2)
with col3:
    خيارات_التخصصي = sub_to_spec.get(تصنيف_فرعي, [])
    تصنيف_تخصصي = st.selectbox("التصنيف التخصصي", خيارات_التخصصي)
with col4:
    البلدية = st.selectbox("البلدية", بلديات)

# الوقت الحالي
الآن = datetime.now()
شهر = الآن.month
يوم_الاسبوع = الآن.weekday()
ساعة = الآن.hour

# تحديد إذا البلاغ ثقيل
ثقيل = 1 if تصنيف_رئيسي in ['الطرق والأرصفة', 'المباني والتعديات'] else 0

# زر التنبؤ
if st.button("عرض المدة التقديرية", use_container_width=True):
    try:
        input_dict = {
            'البلدية': label_encoders['البلدية'].transform([البلدية])[0],
            'التصنيف الرئيسى': label_encoders['التصنيف الرئيسى'].transform([تصنيف_رئيسي])[0],
            'التصنيف الفرعي': label_encoders['التصنيف الفرعي'].transform([تصنيف_فرعي])[0],
            'التصنيف التخصصي': label_encoders['التصنيف التخصصي'].transform([تصنيف_تخصصي])[0],
            'شهر_البلاغ': شهر,
            'يوم_في_الأسبوع': يوم_الاسبوع,
            'ساعة_البلاغ': ساعة,
            'تصنيف_ثقيل': ثقيل
        }

        df_input = pd.DataFrame([input_dict])
        df_input = df_input[features]

        prediction = model.predict(df_input)[0]
        st.success(f"وفقًا للبيانات السابقة، المدة المتوقعة لمعالجة هذا البلاغ: {round(prediction, 2)} يوم")

        st.markdown("""
تم إنشاء هذا التقدير باستخدام تقنيات الذكاء الاصطناعي، وقد يختلف الوقت الفعلي حسب الظروف التشغيلية.
""")
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
