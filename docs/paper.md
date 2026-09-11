# Технология однобитного представления языковых моделей: программная реализация и оценка качества

Заготовка статьи, начата 09.09.2026. Сейчас готова аннотация; текст статьи пишется.
Публикация подается как работа о программной разработке, а не как исследование:
предъявляется технология и ее реализация, а рецензенту показывают состав комплекса,
воспроизводимость и результаты эксплуатации.

**Состояние:** аннотация написана без числовых утверждений - обучение еще не
проводилось. Цифры добавляются одной фразой после первого прогона.

## Аннотация

Нейросетевые языковые модели занимают много памяти, и один из способов ее
сократить - хранить каждый вес одним битом. Простое округление весов до знака для
этого не годится: качество модели разрушается. Пересчет весов по небольшой
выборке текста тоже не помогает - подобранные веса ведут себя хорошо на этой
выборке и плохо на любой другой. Разбор уже опубликованных однобитных моделей
показывает, что их не пересчитывают, а обучают заново.

В работе предложен и реализован такой способ обучения: однобитная модель учится
повторять ответы обычной полноточной модели, оставаясь однобитной на каждом шаге
обучения. Описан программный комплекс, выполняющий весь цикл: подготовку
обучающего текста, обучение с продолжением после остановки, сравнение результата
с исходной моделью и с простым округлением, сохранение готовой модели. Комплекс
работает без доступа к сети, от одной видеокарты до вычислительного кластера, и
распространяется с открытым исходным кодом. Приводятся оценка качества полученных
моделей и требования к вычислительным ресурсам.

## Ключевые слова

однобитное квантование; бинарные нейронные сети; дистилляция знаний; сжатие
нейросетевых моделей; большие языковые модели

## Abstract

Neural language models take up a great deal of memory, and one way to reduce it is
to store each weight in a single bit. Simply rounding weights to their sign does
not work: model quality collapses. Recomputing the weights against a small sample
of text does not help either - the resulting weights behave well on that sample and
poorly on any other. An examination of published one-bit models shows that they are
not recomputed but retrained.

This paper proposes and implements such a training method: a one-bit model learns
to reproduce the answers of an ordinary full-precision model while staying one-bit
at every training step. A software toolkit covering the whole cycle is described:
preparing the training text, training that resumes after an interruption, comparing
the result against both the original model and plain rounding, and saving the
finished model. The toolkit runs without network access, scales from a single
graphics card to a compute cluster, and is released as open source. The quality of
the resulting models and the computational requirements are reported.

## Keywords

one-bit quantization; binary neural networks; knowledge distillation; model
compression; large language models

## Перед подачей

Числовых утверждений в аннотации нет намеренно: обучение еще не проводилось.
После первого прогона в последний абзац добавляется одна фраза с итоговыми
значениями, остальной текст не меняется.
