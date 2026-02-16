---
title: "Brain Tumor Classification Using ResNet"
author: "Ayman Alaqrabawi"
date: "9/2/2026"
output:
  html_document:
    toc: true
    toc_depth: 3
  ---

# 1. Introduction

في هذا المشروع، ركزنا على **تصنيف أورام الدماغ من صور الرنين المغناطيسي (MRI)** لتسهيل التشخيص المبكر وتحسين خطط العلاج. الهدف الأساسي كان بناء نموذج قادر على التمييز بين عدة فئات من الأورام بدقة عالية.

---

# 2. Data Preparation

- **تنظيف البيانات:** التأكد من أن جميع الصور بصيغة صحيحة وحجم موحد.  
- **Normalization:** تحويل قيم البكسل إلى نطاق [0,1] لتسهيل تدريب الشبكة ومنع مشاكل تلاشي التدرجات.  
- **Data Augmentation:** استخدام تقنيات مثل التدوير، التكبير، النقل الأفقي/العمودي لعلاج مشكلة **overfitting** وزيادة تنوع البيانات.  
- **Batch Size:** تم استخدام 32 صورة لكل دفعة.  
- **تقسيم البيانات:** مجموعة تدريب ومجموعة اختبار، كل مجموعة تحتوي على صور موزعة على الفئات الأربع.

---

# 3. Model Architecture

تم استخدام **ResNet50** كهيكلية أساسية للشبكة:

- **Residual (Skip) Connections:** تسمح بإضافة مدخل الطبقة مباشرة لمخرجها، مما يحافظ على التدرجات خلال الشبكات العميقة ويحل مشكلة **vanishing gradient**.  
- **Transfer Learning:** استخدمنا أوزان ImageNet مسبقة التدريب، مما يتيح للنموذج الاستفادة من الميزات البصرية العامة مثل الحواف والأشكال، ويقلل الحاجة لمجموعات بيانات ضخمة.  
- **Fine-Tuning:** تعديل الطبقات العميقة الأخيرة لتتكيف مع خصائص صور أورام الدماغ، ما يحسن قدرة النموذج على التعرف على الأنماط الخاصة بالمهمة.

تمت إضافة:

- **Global Average Pooling:** لتقليل حجم المخرجات وتحويلها إلى تمثيل خطي مناسب للطبقات الكثيفة.  
- **Batch Normalization:** لتحسين استقرار التدريب وتسريع التقارب.  
- **Dense Layers مع Dropout:** لتعزيز التعلم غير الخطي وتقليل الإفراط في التعلم.

---

# 4. Training Strategy

- تم تدريب النموذج على **مجموعة التدريب** مع **عدد Epochs** مناسب للوصول لأفضل دقة.  
- تم استخدام **categorical crossentropy** كدالة خسارة لأنها مناسبة لمهام التصنيف المتعدد.  
- استخدمنا **Adam optimizer** مع معدل تعلم منخفض لضبط أوزان الشبكة بدقة أثناء Fine-Tuning.  
- تم تقييم الأداء على **مجموعة الاختبار** بشكل دوري لتجنب الإفراط في التعلم.

---

# 5. Evaluation

- استخدمنا **التنبؤ على بيانات الاختبار** للحصول على الفئات المتوقعة.  
- تم حساب **Accuracy** كمقياس رئيسي لأداء النموذج.  
- استخدمنا **confusion matrix و classification report** للحصول على تفاصيل مثل precision، recall، و F1-score لكل فئة.

---

# 6. Summary

- نموذج **ResNet50 مع Transfer Learning** فعال في تصنيف أورام الدماغ من صور MRI.  
- **Normalization و Data Augmentation** ساعدت على تقليل overfitting وتحسين أداء النموذج.  
- Fine-tuning الطبقات العميقة مكن النموذج من التعلم من خصائص البيانات الطبية الخاصة بالمهمة.  
- النموذج قادر على تصنيف الأورام بدقة جيدة مع إمكانية تحسين الأداء بالمزيد من البيانات أو التدريب لفترة أطول.

---

# 7. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)  
2. Tan, C., Sun, F., Kong, T., Zhang, W., Yang, C., & Liu, C. (2018). *A Survey on Deep Transfer Learning*. [https://arxiv.org/abs/1808.01974](https://arxiv.org/abs/1808.01974)  
3. Keras ResNet Documentation: [https://keras.io/api/applications/resnet/](https://keras.io/api/applications/resnet/)  
4. ImageNet Large Scale Visual Recognition Challenge (ILSVRC): [https://www.image-net.org/](https://www.image-net.org/)  
5. Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(60). [https://doi.org/10.1186/s40537-019-0197-0](https://doi.org/10.1186/s40537-019-0197-0)
