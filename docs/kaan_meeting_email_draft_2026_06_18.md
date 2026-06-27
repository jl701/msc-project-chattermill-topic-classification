# Email Draft To Dr Kaan Aksit

Subject: MSc project update and pre-read for next supervision meeting

Dear Kaan,

I hope you are well. Ahead of our meeting next week, I wanted to send a short update on my MSc project with Chattermill and share a brief pre-read.

As a quick reminder, the project is about automatically assigning business topic labels to customer feedback. For example, a review might mention both app usability and customer support, and the model should assign the relevant topic labels, together with the sentiment expressed about each topic.

Since our previous meeting, I have set up a reproducible experimental pipeline using the public FABSA customer-review dataset, implemented several baselines, and started shaping the project around two main questions: whether models generalise to feedback from unseen companies, and whether they can handle new topic labels supplied at inference time.

The early results suggest that the fixed-label version of the task is reasonably well handled by a DistilBERT baseline, but the new-topic setting is substantially harder and is likely the more interesting research direction. I have also prepared a Qwen3-4B candidate-label prompting and data-preparation workflow, but Qwen has not yet been fully fine-tuned for the main held-out-topic evaluation.

I would be grateful if you could skim the attached brief before the meeting. In particular, I would appreciate your advice on whether the current evaluation framing is academically sound, whether the held-out aspect protocol is defensible, and how best to position the project contribution for the dissertation.

Best regards,
Lester
