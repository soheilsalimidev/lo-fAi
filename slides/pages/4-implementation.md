---
---
# پیاده سازی

- **جمع آوری داده**
- **تبدیل داده ها به متن**
- **آموزش مدل ها**
- **ارزیابی عملکرد مدل ها**
- **نحویه ساخت موسیقی نهایی**

---
---
## جمع آوری داده
### مدل پیانو
|                 | IrishMMD Dataset <a href="/30">\[5\]</a> |
| --------------- | ---------------- |
| کل فایل ها      | 216,284          |
| فایل های آموزش  | 214,208 (99%)    |
| فایل اعتبارسنجی | 2,076 (1%)       |

---
---

### مدل دارم

<p class="text-center">
Expanded Groove MIDI Dataset <a href="/30">[6]</a>
</p>

| بخش        | تعداد کل توالی‌ها | مدت زمان (ساعت) |
| ---------- | ---------------- | --------------- |
| آموزشی     | 35,217           | 341.4           |
| اعتبارسنجی | 5,031            | 52.2            |
| کل         | 45,537           | 444.5           |



---
---
## تبدیل MIDI به متن

<Caption caption="نحویه تبدیل MIDI به متن">
  <div class="flex max-h-full ">
   <img src="/Untitled 1.png" class="max-h-full object-contain"/>
  </div>
</Caption>

---
layout: "two-cols-gl"
---


```py {all|2|5|9-12}
def format_wait_token(wait):
  return f"t{wait}"

def format_note_token(note, velocity_bin):
  return f"{note:x}:{velocity_bin:x}"

def velocity_to_bin(velocity):
  velocity = min(max(velocity, 0), 128 - 1)
  return ceil((128 *
   ((0.33 ** (velocity /128) - 1)
    / (0.33 - 1)) / 16))
```
<template v-slot:right>

```py {all|2|3|4-5|6-9|10|all}
def convert_midi_to_str(mid, cfg, augment=None):
  output_list = []
  for msg in mid.tracks[0]:
    delta_time = \
      mido.tick2second(msg.time, tempo) * 1000.0
    if msg.is_meta and msg.type == "set_tempo":
      tempo *= augment.time_stretch_factor
    elif msg.type in ("note_on", "note_off"):
      handle_note(msg.channel, msg.velocity, msg.note)
  flush_token_data_buffer(output_list)
  return output_list

def handle_note(channel, velocity, note):
  if velocity == 0:
    # handle note off
  else:
    # handle note on
    consume_note_program_data(channel, note, velocity)
```
</template>
$$
\left\lceil \frac{ 128 \cdot \left( \frac{ 0.33^{\left( \frac{\text{velocity}}{128} \right)} - 1 }{ 0.33 - 1 } \right) }{ 16 } \right\rceil
$$

<!-- Normalize velocity: To ensure a consistent range for the calculation.
Apply an exponential decay: To produce a decreasing output as velocity increases.
Scale and quantize: To map the result to a discrete range of values. -->

---
---

## آموزش مدل ها
ما از معماری RWKV-6.0 <a href="/30">\[7\]</a> با 20 لایه و Embedding برابر با 512 استفاده کردیم. طول زمینه مدل 512 است. نرخ یادگیری اولیه و نهایی به ترتیب 6e-4 و 6e-5 است و از تابع آنتروپی متقاطع برای محاسبه خطا استفاده می‌شود. پارامتر head_size_a برابر با 64 انتخاب شده تا مدل بتواند به تعداد بیشتری از عناصر ورودی توجه کند. از کتابخانه DeepSpeed نیز استفاده شده که شامل بهینه‌ساز Adam و زمان‌بند نرخ یادگیری کاهش گرم‌شونده است. همچنین، دقت مختلط bfloat16 یا float16 برای بهبود سرعت آموزش فعال شده است.

---
---

## بررسی یادگیری مدل

<Caption caption="نمودار تابع خطا آموزش مدل های درام و پیانو">
  <img src="/loss-dr.png" class="  max-w-1/2 object-fit"/>
  <img src="/loss-pi.png" class=" max-w-1/2 object-fit"/>
</Caption>


---
---
## ارزیابی عملکرد مدل
برخی از معیار های موسیقی <a href="/30">\[8\]</a>

- **هماهنگی ریتم**: هماهنگی ریتم یک متریک است که به بررسی نوسان یا یکنواختی طول نت ها در یک قطعه موسیقی می پردازد. این متریک نشان می دهد که قطعه های موسیقی چه میزان یکنواختی و یا چه میزان نوسانی دارند.

- **شباهت ملودیک**: شباهت ملودی، یک روش برای اندازه گیری شباهت بین دو ملودی بر اساس ترتیب نت‌هایشان است. این متد ساده و مستقیماً از نسبت نت‌های مشابه دو ملودی برای محاسبه شباهت استفاده می‌کند.

- **انسجام هارمونیک**: اندازه‌گیری میزان سازگاری و انسجام هارمونی‌ها و پیشرفت‌های آکورد در یک قطعه موسیقی با کلید یا مرکز تونال زیرین، با در نظر گرفتن عواملی مانند عملکرد آکورد، هدایت صدا و حل هارمونیک.

<!--
**هماهنگی ریتم**: باید در یک سرعت نوت ها اجرا شود و روند تغییر سرعت حفط شود
**شباهت ملودیک**: وقتی نود هایی که پلی میشند یکی هستند و یه حسی خاصی را همیشه منتقل می کنند
**انسجام هارمونیک**: وقتی چند نوتی که با هم پلی میشند در کل موسیقی یکنواخت باشه نوع اهنگ مقلا از شاد به غمیگین نرع
-->

---
---
###  ارزیابی عملکرد مدل پیانو

| متد                          | مقدار |
| ---------------------------- | ----- |
| هماهنگی ریتم                 | 0.34  |
| هماهنگی هارمونیک (ملایمت)    | 0.87  |
| هماهنگی هارمونیک (نا‌هم‌خوانی) | 0.13  |
| شباهت ملودیک                 | 0.09  |

---
---

## ساخت موسیقی نهایی

<Caption caption="روند ساخت موسیقی نهایی">
    <img src="/Untitled 2.png" class="object-contain scale-110 mt-5"/>
</Caption>

---

```py{all|10-13|1-5|15-18}
def midiToWav():
    # Load soundfont and synthesize MIDI to WAV
    synth = tinysoundfont.Synth()
    synth.midi_load(midi)
    return synth.tobytes()

class GenMusic:
    def __init__(self, data, dataDrum):
        # Generate MIDI and convert to audio segments
        tempo = random.randint(14, 18) * 4 * 3
        self.pianoRoll = AudioSegment(midiToWav("./OmegaGMGS2.sf2",data))
        self.fill = AudioSegment.from_file(random.choice(os.listdir("./loops/vinyl")))
        self.drum = AudioSegment(midiToWav("./FluidR3_GM.sf2",dataDrum))

    def mix_lines(self, music_len=60):
        return music.fade_out(2000), \
        self.RandRoll.overlay(self.fill, position=5).overlay(self.drum, position=-5) \
         * (music_len // self.RandRoll.duration_seconds).fade_out(2000)
```

<!--
temp from 168 to 216
-->

---
---
# خط لوله متن به Lo-fi

<Caption caption="روند خط لوله" cla="-mt-15">
    <img src="/123.crop.png"  class="object-contain -mt-15" />
</Caption>

---
layout: "default"
---

## ارزیابی خط لوله متن به Lo-fi
برای ارزیابی این خط لوله ما از پنچ نفر نظرسنجی کردیم. نتایج به شکل زیر است:

| سوال                   | پاسخ                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| وضوح صدای تولید شده    | 20\% (یک نفر) (خیلی بلند بودن یک ساز و خیلی آرام بودن ساز دیگر)آن را ضعیف ارزیابی کردند و به ناهمخوانی درجه صدا ها اشاره کردند. |
| کیفیت موسیقی تولید شده | 40\% (دو نفر) احساس کردند که صدا عمق و احساس کافی ندارد و آن را تخت توصیف کردند.                                                |
| تطابق با ورودی‌ها       | همه نظر دهندگان به عدم تطابق بین انتظارات و خروجی اشاره کردند.                                                                  |


---
---
# نمایش پروژه

<SlidevVideo controls>
  <source src="/2024-09-17 19-36-04.mp4"  type="video/mp4"/>
  <p>
    Your browser does not support videos. You may download it
    <a href="/2024-09-17 19-36-04.mp4">here</a>.
  </p>
</SlidevVideo>
