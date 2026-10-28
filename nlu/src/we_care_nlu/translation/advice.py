"""General guidance when no guideline sentence is found.

These lines are not a diagnosis. They point to We Care's own articles and phone line.
Each entry is (Amharic, English).
"""

from __future__ import annotations

_PHONE = "9394"
_BOOK = "https://wecare.et/booking"
_SITE = "https://wecare.et"

_REFERRAL_AM = (
    f"ይህ አጠቃላይ መረጃ ነው። የሐኪም ምርመራን አይተካም። "
    f"ለበለጠ መረጃ {_PHONE} ይደውሉ ወይም {_BOOK} ላይ ቀጠሮ ይያዙ።"
)
_REFERRAL_EN = (
    f"This is general information. It does not replace a clinical exam. "
    f"For more information call {_PHONE} or book at {_BOOK}."
)


def _pair(am: str, en: str) -> tuple[str, str]:
    return f"{am}\n{_REFERRAL_AM}", f"{en}\n{_REFERRAL_EN}"


_ADVICE: dict[str, tuple[str, str]] = {
    "family_planning_methods": _pair(
        "የእርግዝና መከላከያ ዘዴዎች በርካታ አማራጮች አሉ። የመከላከያ ዘዴዎች ኮንዶምን ያካትታሉ። "
        "የሆርሞን ዘዴዎች ክኒን፣ መርፌ እና ኢምፕላንት ናቸው። በማሕፀን ውስጥ የሚገባ መከላከያ (አይዩሲዲ) ለረጅም ጊዜ ይቆያል። "
        "የአስቸኳይ መከላከያ እና ቋሚ ዘዴዎችም አሉ። የሚስማማው ዘዴ በእያንዳንዱ ሰው ሁኔታ ይለያያል።",
        "Contraceptive options include barrier methods such as condoms; hormonal methods such as pills, injectables, and implants; "
        "intrauterine devices (IUCD); emergency contraception; and permanent methods. "
        "The suitable method depends on the person. "
        f"More information: {_SITE}",
    ),
    "definition": _pair(
        "ስለ አንድ የእርግዝና መከላከያ ዘዴ አጭር መግለጫ በመመሪያው ውስጥ አላገኘሁም። "
        "ዘዴው ምን እንደሆነ እና ለእርስዎ ይስማማ እንደሆነ ከጤና ባለሙያ ጋር ማረጋገጥ ይሻላል። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "I could not find a short description of that method in the guidelines. "
        "Check with a health worker whether it suits you. "
        f"More information: {_SITE}",
    ),
    "duration": _pair(
        "አንድ ዘዴ የሚቆይበት ጊዜ በዘዴው ይለያያል። ለምሳሌ ኢምፕላንት ለበርካታ ዓመታት ሊቆይ ይችላል፣ ክኒን ግን በየቀኑ ይወሰዳል። "
        "ለእርስዎ የተወሰነውን ጊዜ ከጤና ባለሙያ ያረጋግጡ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "How long a method lasts depends on the method. An implant can last several years, while a pill is taken every day. "
        "Confirm the timing for you with a health worker. "
        f"More information: {_SITE}",
    ),
    "side_effects": _pair(
        "የጎንዮሽ ጉዳት በዘዴው ይለያያል። የደም መፍሰስ ለውጥ፣ ራስ ምታት ወይም የስሜት ለውጥ ሊኖር ይችላል። "
        "ከባድ ወይም የሚያሳስብ ምልክት ካለ ወደ ጤና ተቋም ይሂዱ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Side effects differ by method. Bleeding changes, headache, or mood changes can happen. "
        "Go to a health facility for a severe or worrying symptom. "
        f"More information: {_SITE}",
    ),
    "contraindications": _pair(
        "አንዳንድ ዘዴዎች ለነፍሰ ጡር፣ ለሚያጠቡ፣ ወይም ለተወሰኑ የጤና ሁኔታዎች አይመከሩም። "
        "ከመጀመርዎ በፊት ከጤና ባለሙያ ጋር ያረጋግጡ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Some methods are not advised in pregnancy, while breastfeeding, or with certain health conditions. "
        "Check with a health worker before starting. "
        f"More information: {_SITE}",
    ),
    "eligibility": _pair(
        "አንድ ዘዴ ለእርስዎ ይስማማ እንደሆነ በእድሜ፣ በጤና ሁኔታ እና በሌሎች መድኃኒቶች ይወሰናል። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Whether a method suits you depends on age, health, and other medicines. "
        f"More information: {_SITE}",
    ),
    "missed_dose": _pair(
        "ክኒን ከተረሳ የሚደረገው እርምጃ በክኒኑ ዓይነት እና በተረሳበት ጊዜ ይለያያል። "
        "ተጨማሪ ክኒን ከመውሰድዎ በፊት መመሪያውን ወይም የጤና ባለሙያን ይጠይቁ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "What to do after a missed pill depends on the pill and how late it is. "
        "Check the instructions or a health worker before taking extra pills. "
        f"More information: {_SITE}",
    ),
    "access": _pair(
        "የእርግዝና መከላከያ በጤና ጣቢያ፣ በሆስፒታል እና በአንዳንድ ፋርማሲዎች ይገኛል። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Contraception is available at health centres, hospitals, and some pharmacies. "
        f"More information: {_SITE}",
    ),
    "how_it_works": _pair(
        "ዘዴዎች በተለያየ መንገድ ይሠራሉ። አንዳንዶቹ እንቁላል እንዳይወጣ ይከላከላሉ፣ ሌሎች ደግሞ ወደ ማሕፀን የሚደረገውን ጉዞ ይከላከላሉ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Methods work in different ways. Some stop an egg from being released. Others block the path to the womb. "
        f"More information: {_SITE}",
    ),
    "method_comparison": _pair(
        "ዘዴዎች በሚቆዩበት ጊዜ፣ በጎንዮሽ ጉዳት እና በምን ያህል እንደሚከላከሉ ይለያያሉ። የሚሻለው ዘዴ ለእያንዳንዱ ሰው አንድ አይደለም። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Methods differ in how long they last, their side effects, and how well they prevent pregnancy. "
        "There is no single best method for everyone. "
        f"More information: {_SITE}",
    ),
    "pregnancy_breastfeeding": _pair(
        "በእርግዝና እና ጡት በማጥባት ጊዜ የሚስማሙ ዘዴዎች አሉ፣ ሁሉም ግን አይመከሩም። ከጤና ባለሙያ ጋር ይምረጡ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Some methods can be used in pregnancy or while breastfeeding. Not all of them can. Choose with a health worker. "
        f"More information: {_SITE}",
    ),
    "symptoms": _pair(
        "ምልክት ብቻውን በሽታ አይወስንም። የሚያሳስብ ሕመም፣ ቁስለት ወይም ያልተለመደ ፈሳሽ ካለ በጤና ተቋም ይመረመሩ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "A symptom by itself does not decide an illness. Go to a health facility for worrying pain, a sore, or unusual discharge. "
        f"More information: {_SITE}",
    ),
    "causes": _pair(
        "አንድ ምልክት ብዙ ምክንያቶች ሊኖሩት ይችላል። ምክንያቱን ለማወቅ ምርመራ ያስፈልጋል። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "One symptom can have several causes. An exam is needed to know which one. "
        f"More information: {_SITE}",
    ),
    "prevention": _pair(
        "ኮንዶም አንዳንድ ኢንፌክሽኖችን እና እርግዝናን ለመቀነስ ይረዳል። ሌሎች የመከላከያ መንገዶች በጉዳዩ ይለያያሉ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Condoms reduce some infections and pregnancy. Other ways to prevent illness depend on the condition. "
        f"More information: {_SITE}",
    ),
    "treatment": _pair(
        "ሕክምናው በምርመራው ይወሰናል። መድኃኒት ከመጀመርዎ በፊት በጤና ተቋም ይመረመሩ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "Treatment depends on the diagnosis. Be examined at a health facility before starting medicine. "
        f"More information: {_SITE}",
    ),
    "medication": _pair(
        "የመድኃኒት መጠን እና ጊዜ በዓይነቱ ይለያያል። የታዘዘውን መጠን ይከተሉ፣ ወይም ከጤና ባለሙያ ይጠይቁ። "
        f"ተጨማሪ መረጃ፦ {_SITE}",
        "The dose and timing depend on the medicine. Follow what was prescribed, or ask a health worker. "
        f"More information: {_SITE}",
    ),
    "vaginal_discharge": _pair(
        "ከሴት ብልት የሚወጣ ፈሳሽ በተለምዶ የሚኖር እና አስፈላጊ ፈሳሽ ነው። በወር አበባ ዑደት ሊለዋወጥ ይችላል። "
        "መጥፎ ሽታ፣ ማሳከክ፣ ወይም የቀለም ለውጥ ካለ ኢንፌክሽን ሊሆን ይችላል። ይህ ምርመራ አይደለም። "
        "መደበኛ የብልት ፈሳሽ፦ https://wecare.et/articles/420\n"
        "በወር አበባ ወቅት የሚኖር ለውጥ፦ https://wecare.et/articles/421",
        "Vaginal discharge is a normal and necessary fluid. It can change across the menstrual cycle. "
        "A bad smell, itching, or a change in colour can be a sign of infection. This is not a diagnosis. "
        "Vaginal discharge: https://wecare.et/articles/420\n"
        "Changes during menstruation: https://wecare.et/articles/421",
    ),
    "hiv": _pair(
        "ኤች.አይ.ቪ ብዙ ጊዜ መጀመሪያ ላይ ምልክት አያሳይም። ምልክት ብቻ ሁኔታውን አያሳውቅም። "
        "ምርመራ በጤና ተቋም ይደረጋል። ከተጋላጭነት በኋላ በ72 ሰዓት ውስጥ የሚወሰድ መድኃኒት (PEP) ካስፈለገ ወዲያውኑ ይጠይቁ። "
        "የኤች.አይ.ቪ ምርመራ፦ https://wecare.et/articles/410",
        "HIV often has no symptoms at first, so symptoms alone cannot show your status. "
        "Testing is done at a health facility. If you may need PEP, the medicine taken within 72 hours of exposure, ask immediately. "
        "HIV testing: https://wecare.et/articles/410",
    ),
    "menstruation": _pair(
        "የወር አበባ ዑደት በሰው ይለያያል። ዘግይቶ መምጣት፣ በጣም ማሳመም፣ ወይም ከተለመደው በላይ መፍሰስ ምርመራ ሊፈልግ ይችላል። "
        "ስለ ወር አበባ፦ https://wecare.et/articles/310",
        "Menstrual cycles differ from person to person. A delayed period, severe pain, or bleeding that is heavier than usual may need an exam. "
        "About menstruation: https://wecare.et/articles/310",
    ),
    "pregnancy": _pair(
        "የእርግዝና ምልክት ካለ የእርግዝና ምርመራ ይመከራል። የቅድመ ወሊድ ክትትል የሽንት ምርመራ፣ የደም ምርመራ እና አልትራሳውንድ ሊያካትት ይችላል። ይህ ምርመራ አይደለም። "
        "ስለ እርግዝና፦ https://wecare.et/articles/416\n"
        "እርግዝና እና አባላዘር፦ https://wecare.et/articles/261",
        "If you have pregnancy symptoms, a pregnancy test is advised. Antenatal care can include a urine test, blood tests, and ultrasound. This is not a diagnosis. "
        "About pregnancy: https://wecare.et/articles/416\n"
        "Pregnancy and STIs: https://wecare.et/articles/261",
    ),
    "sti": _pair(
        "አባላዘር በሽታ ምልክት ላይኖረው ይችላል። ያልተለመደ ፈሳሽ፣ ሽንት ሲሸኑ ማቃጠል፣ ወይም የብልት ቁስለት ካለ በጤና ተቋም ይመረመሩ። ይህ ምርመራ አይደለም። "
        "የብልት ፈሳሽ፦ https://wecare.et/articles/421\n"
        "የብልት ቁስለት እና ሄርፒስ፦ https://wecare.et/articles/405",
        "A sexually transmitted infection can have no symptoms. Go to a health facility for unusual discharge, burning when you urinate, or a genital sore. This is not a diagnosis. "
        "Vaginal discharge: https://wecare.et/articles/421\n"
        "Genital sores and herpes: https://wecare.et/articles/405",
    ),
    "genital_ulcer": _pair(
        "የብልት ቁስለት በሴት ብልት፣ በወንድ ብልት፣ በፊንጢጣ ወይም አንዳንዴ በአፍ አካባቢ የሚታይ ቁስል ወይም ሕመም ነው። "
        "በብዛት የሚጠቀሱ ምክንያቶች የብልት ሄርፒስ፣ ቂጥኝ እና ቻንክሮይድ ናቸው። ይህ ምርመራ አይደለም። "
        "የብልት ቁስለት፦ https://wecare.et/articles/135\n"
        "ሄርፒስ፦ https://wecare.et/articles/405\n"
        "ቂጥኝ፦ https://wecare.et/articles/389",
        "A genital ulcer is a sore or painful spot on the vulva, penis, anus, or sometimes around the mouth. "
        "Commonly named causes are genital herpes, syphilis, and chancroid. This is not a diagnosis. "
        "Genital ulcer: https://wecare.et/articles/135\n"
        "Herpes: https://wecare.et/articles/405\n"
        "Syphilis: https://wecare.et/articles/389",
    ),
    "general_medical_info": _pair(
        "ለዚህ ጥያቄ በመመሪያው ውስጥ የሚዛመድ ዓረፍተ ነገር አላገኘሁም። "
        f"አጠቃላይ መረጃ በ {_SITE} ላይ አለ።",
        "I could not find a matching sentence in the guidelines for this question. "
        f"General information is on {_SITE}.",
    ),
}


def intent_advice(intent: str | None) -> tuple[str, str] | None:
    if not intent:
        return None
    return _ADVICE.get(intent)
