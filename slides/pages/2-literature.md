---
layout: "new-def"
---

#  ادبیات موضوع
## RWKV: Reinventing RNNs for the Transformer Era

مدل  RWKV <a href="/30">\[1\]</a> یک مدل پیشرفته است که کارایی شبکه‌های عصبی برگشت‌پذیر را با قدرت توجه به خود ترکیب می‌کند و امکان انجام سریع و مؤثر وظایف دنباله به دنباله را فراهم می‌سازد. 

---
layout: "default"
---

### RWKV معماری

<Caption caption="معماری RWKV, گرفته شده از [1]">
    <img src="/RWKV-arch.png" class="object-contain scale-90"/>
    <img src="/x2.png" class=" w-50"/>
</Caption>

---
layout: "new-def"
---

## MIDI

MIDI <a href="/30">\[2\]</a> یک فرمت مبتنی بر متن است که موسیقی را به صورت الکترونیکی نمایش می‌دهد. این فرمت شامل اطلاعات نوت‌ها مانند ارتفاع صدا، مدت زمان و تمپو است. هیچ داده صوتی ندارد و فقط دستورالعمل‌هایی برای سازها ارائه می‌دهد. سبک و آسان برای ویرایش است. ورودی کلیدی برای مدل ما جهت تولید موسیقی لو-فای است.

<!-- Represents music as notes and events • No audio data, only instructions • Lightweight and easy to edit • Crucial input for our model • Wide range of musical styles -->

---
---

# پروژه های پیشن
## jacbz/Lofi

<div class="text-center mt-2 italic bold">
ML-supported lo-fi music generator Based on VAE <a href="/30">[3]</a>
</div>

---
layout: "default"
---
## مشکلات jacbz/Lofi


1. **مدیریت ناکافی وابستگی‌های ترتیبی**: طراحی VEA عمدتاً بر پردازش تصویر متمرکز است و این امر آن را برای وظایف ترتیبی مانند تولید موسیقی کمتر مؤثر می‌کند.
2. **عدم وجود مکانیزم توجه**: مکانیزم توجه VEA ساختار حافظه صریحی ارائه نمی‌دهد که برای ذخیره و بازیابی اطلاعات در وظایف ترتیبی حیاتی است.
3. **معماری غیرقابل انعطاف**: معماری VEA به راحتی قابل تطبیق با وظایف مختلف نیست و این امر تنظیم دقیق آن برای تولید موسیقی را دشوار می‌کند.
4. **عدم کنترل صریح بر خروجی**: خروجی VEA توسط latent space تعیین می‌شود که کنترل دنباله خروجی را چالش‌برانگیز می‌کند.

