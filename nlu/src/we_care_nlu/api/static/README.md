# We Care chat UI

Small browser chat for the NLU service (`we-care-nlu serve`).

## Files

| File | Role |
|------|------|
| `index.html` | Page shell: header, message log, typing box |
| `styles.css` | Layout and look |
| `app.js` | Talks to the API and draws messages |

Served from the FastAPI app: `/` → HTML, `/static/...` → CSS/JS.

## What the page does

1. On load, polls `GET /health` until the model + knowledge base are ready.
2. User types a question (Amharic or English) and clicks **ላክ** / presses Enter.
3. The page calls `POST /nlp/reply` with the text and a small conversation `context`.
4. The answer is shown in the chat log (English ask → prefer English text; Amharic ask → prefer Amharic).

## Contact line

The welcome card (above the typing box) shows hotline **9394** and booking **https://wecare.et/booking**.  
Answer text may also include the booking URL on “no match” fallbacks; other article source links are stripped in the UI.

## Conversation context

`app.js` keeps `context` between turns (`last_topic`, `last_method`, …) so follow-ups like “ሲያመኝ ምን ላድርግ?” stay on the right medical topic. Clinical topics such as menstruation clear a leftover contraceptive method so old FP answers do not leak in.
