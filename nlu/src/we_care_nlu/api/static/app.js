/**
 * We Care chat UI — browser client for the NLU API.
 *
 * Flow:
 *   1. waitUntilEngineReady()  → poll /health
 *   2. user types + Send       → POST /nlp/reply
 *   3. showAnswer()            → draw the reply in #log
 *   4. updateConversationContext() so follow-up questions keep topic/method memory
 *
 * See README.md in this folder for a short overview.
 */

(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // Fixed copy (greetings, clarifications, brand line)
  // ---------------------------------------------------------------------------

  const COPY = {
    brandTag: "የሥርዓተ-ተዋልዶ ጤና መረጃ · Reproductive health information",
    talk: {
      greeting: "እንኳን ደህና መጡ",
      conversation_end: "ደህና ሁኑ። ሌላ ጥያቄ ካለዎት እዚህ ይመለሱ።",
      out_of_domain:
        "አሁን የምሰጠው መረጃ ስለ የሥርዓተ-ተዋልዶ ጤና ነው። እባክዎ ስለ የሥርዓተ-ተዋልዶ ጤና ይጠይቁ።",
    },
    clarify: {
      which_method: "የትኛውን የእርግዝና መከላከያ ዘዴ ማለትዎ ነው?",
    },
  };

  /** Topics where we must forget a previous contraceptive method / breastfeeding flag. */
  const TOPICS_THAT_CLEAR_METHOD = new Set([
    "menstruation",
    "irregular_menses",
    "endometriosis",
    "menopause",
    "cryptic_pregnancy",
    "vaginal_discharge",
    "genital_ulcer",
    "uti",
    "cervical_cancer",
    "breast_cancer",
    "mental_health",
  ]);

  // ---------------------------------------------------------------------------
  // DOM + session state
  // ---------------------------------------------------------------------------

  const logEl = document.getElementById("log");
  const inputEl = document.getElementById("q");
  const sendBtn = document.getElementById("go");
  const statusTag = document.getElementById("statusTag");

  const conversationId = crypto.randomUUID();
  let conversationContext = null;
  let engineReady = false;
  let welcomeCard = null;

  // ---------------------------------------------------------------------------
  // Small DOM helpers
  // ---------------------------------------------------------------------------

  function htmlToNode(html) {
    const wrap = document.createElement("div");
    wrap.innerHTML = html;
    return wrap.firstElementChild;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[ch]
    );
  }

  function appendToLog(html) {
    const node = htmlToNode(html);
    logEl.appendChild(node);
    logEl.scrollTop = logEl.scrollHeight;
    return node;
  }

  function setStatus(message) {
    if (statusTag) statusTag.textContent = message;
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ---------------------------------------------------------------------------
  // Language / reply text cleanup
  // ---------------------------------------------------------------------------

  /** Prefer English UI when the ask is mostly Latin letters. */
  function askedInEnglish(text) {
    const eth = (String(text).match(/[\u1200-\u137F]/g) || []).length;
    const lat = (String(text).match(/[A-Za-z]/g) || []).length;
    return lat > eth;
  }

  /**
   * Drop article/source URLs from answers.
   * Keep https://wecare.et/booking (used on no-match fallbacks).
   */
  function stripSourceLinks(text) {
    return String(text || "")
      .replace(/https?:\/\/(?!wecare\.et\/booking)\S+/gi, "")
      .replace(/(?:ተጨማሪ(?:\s*መረጃ)?|More(?:\s*information)?)\s*[:፦]\s*/gi, "")
      .replace(/[ \t]+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function linkifyHttp(text) {
    return escapeHtml(text).replace(/https?:\/\/[^\s]+/g, (url) => {
      return `<a href="${url}" target="_blank" rel="noopener">${url}</a>`;
    });
  }

  // ---------------------------------------------------------------------------
  // Rendering answers
  // ---------------------------------------------------------------------------

  function renderAnswerHtml(apiData, userText) {
    const emergency = (apiData.safety || {}).emergency || {};
    if (emergency.is_emergency) {
      return `<p class="emerg">ይህ አስቸኳይ ሊሆን ይችላል። እባክዎ ወደ ቅርብ የጤና ተቋም ይሂዱ።</p>`;
    }

    if (COPY.talk[apiData.intent]) {
      return `<p class="am">${escapeHtml(COPY.talk[apiData.intent])}</p>`;
    }

    if (apiData.requires_clarification) {
      const key = apiData.clarification_question_key;
      const msg = COPY.clarify[key] || "እባክዎ ጥያቄውን ያብራሩ።";
      return `<p class="clarify">${escapeHtml(msg)}</p>`;
    }

    const amharic = apiData.amharic_reply || "";
    const english = apiData.english_reply || "";
    const preferEnglish = askedInEnglish(userText || "");
    const body = stripSourceLinks(preferEnglish ? english || amharic : amharic || english);
    return `<p class="am">${linkifyHttp(body).replace(/\n/g, "<br>")}</p>`;
  }

  // ---------------------------------------------------------------------------
  // Conversation memory sent on the next /nlp/reply call
  // ---------------------------------------------------------------------------

  function updateConversationContext(apiData) {
    const entities = apiData.entities || [];
    const method = entities.find((e) => e.type === "contraceptive_method");
    const topic = apiData.topic || (apiData.topics && apiData.topics[0]) || null;
    const topicKey = String(topic || "").toLowerCase();
    const clearMethod = TOPICS_THAT_CLEAR_METHOD.has(topicKey);

    let kept = entities.map((e) => ({
      type: e.type,
      value: e.value,
      surface: e.surface,
    }));
    if (clearMethod) {
      kept = kept.filter(
        (e) => e.type !== "contraceptive_method" && e.type !== "breastfeeding_status"
      );
    }

    conversationContext = {
      turn_index: ((conversationContext && conversationContext.turn_index) || 0) + 1,
      last_intents: [apiData.intent].concat(apiData.intent_labels || []).slice(0, 6),
      last_entities: kept,
      last_topic: topic || null,
      last_method: clearMethod ? null : (method && method.value) || null,
    };
  }

  // ---------------------------------------------------------------------------
  // Engine readiness (health poll + optional demo KB ingest)
  // ---------------------------------------------------------------------------

  async function waitUntilEngineReady() {
    sendBtn.disabled = true;
    setStatus("ሞዴሉ እየተጫነ ነው… / Loading model…");

    for (let attempt = 0; attempt < 120; attempt++) {
      try {
        const health = await (await fetch("/health")).json();

        if (health.status === "ok") {
          if ((health.chunks || 0) === 0) {
            await fetch("/documents/ingest", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                path: "data/kb/fp_user_kb.json",
                document_version: "demo",
                review_status: "unreviewed",
              }),
            });
          }
          engineReady = true;
          sendBtn.disabled = false;
          setStatus(COPY.brandTag);
          return;
        }

        if (health.status === "error") {
          setStatus(`ስህተት / Error: ${health.error || "load failed"}`);
          return;
        }

        setStatus("ሞዴሉ እየተጫነ ነው… / Loading model and knowledge base…");
      } catch {
        setStatus("ሰርቨሩ አልተገኘም… / Waiting for server…");
      }
      await sleep(2000);
    }

    setStatus("ጭነት አልተጠናቀቀም — ገጹን ያድሱ / Still loading — refresh the page");
  }

  // ---------------------------------------------------------------------------
  // Send a question
  // ---------------------------------------------------------------------------

  async function sendQuestion() {
    const text = inputEl.value.trim();
    if (!text) return;

    if (!engineReady) {
      setStatus("እባክዎ ሞዴሉ እስኪጠናቀቅ ይጠብቁ… / Please wait until the model is ready…");
      return;
    }

    inputEl.value = "";
    appendToLog(`<div class="user">${escapeHtml(text)}</div>`);
    if (welcomeCard) {
      welcomeCard.remove();
      welcomeCard = null;
    }

    sendBtn.disabled = true;
    const englishAsk = askedInEnglish(text);
    const waitingCard = appendToLog(
      `<article class="card"><p class="wait">${
        englishAsk ? "Please wait…" : "እባክዎ ይጠብቁ…"
      }</p></article>`
    );

    try {
      const res = await fetch("/nlp/reply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          conversation_id: conversationId,
          context: conversationContext,
        }),
      });

      if (res.status === 503) {
        engineReady = false;
        waitingCard.innerHTML = `<p class="empty">${
          englishAsk
            ? "Server is still loading. Wait a moment and try again."
            : "ሰርቨሩ እየተጫነ ነው። ትንሽ ጠብቀው እንደገና ይሞክሩ።"
        }</p>`;
        waitUntilEngineReady();
        return;
      }

      const data = await res.json();
      waitingCard.innerHTML = renderAnswerHtml(data, text);
      updateConversationContext(data);
    } catch {
      waitingCard.innerHTML = `<p class="empty">${
        englishAsk
          ? "Could not get an answer. Please try again."
          : "አሁን መልስ ማግኘት አልተቻለም። እባክዎ እንደገና ይሞክሩ።"
      }</p>`;
    } finally {
      if (engineReady) sendBtn.disabled = false;
      inputEl.focus();
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------

  function showWelcomeCard() {
    welcomeCard = appendToLog(`
      <div class="welcome">
        <h2>እንኳን ደህና መጡ</h2>
        <p>ስለ የሥርዓተ-ተዋልዶ ጤና በአማርኛ ወይም በእንግሊዝኛ መጠየቅ ይችላሉ። You can ask in Amharic or English.</p>
        <p class="fine">ይህ መልስ የሐኪም ምክር አይተካም። This is information, not a clinical exam.</p>
        <p class="contact">ለበለጠ መረጃ <a href="tel:9394">9394</a> ይደውሉ ወይም <a href="https://wecare.et/booking" target="_blank" rel="noopener">wecare.et/booking</a> ላይ ቀጠሮ ይያዙ።</p>
      </div>
    `);
  }

  function wireEvents() {
    sendBtn.addEventListener("click", sendQuestion);
    inputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendQuestion();
      }
    });
  }

  showWelcomeCard();
  wireEvents();
  waitUntilEngineReady();
})();
