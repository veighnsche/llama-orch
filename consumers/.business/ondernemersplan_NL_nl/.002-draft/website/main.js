// Minimal progressive enhancement for email capture
(function () {
  const byId = (id) => document.getElementById(id);
  const form = byId('lead-form');
  if (!form) return;

  const email = byId('email');
  const consent = byId('consent');
  const statusEl = byId('form-status');
  const endpoint = document.body?.dataset?.formEndpoint || '';
  const mailTo = document.body?.dataset?.mailTo || '';

  const setStatus = (msg, type = 'info') => {
    statusEl.textContent = msg;
    statusEl.className = type; // style via .info/.ok/.err if desired
  };

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const emailVal = (email.value || '').trim();
    if (!emailVal || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(emailVal)) {
      setStatus('Vul een geldig e‑mailadres in.', 'err');
      email.focus();
      return;
    }
    if (!consent.checked) {
      setStatus('Vink a.u.b. de toestemmingsverklaring aan.', 'err');
      consent.focus();
      return;
    }

    const payload = {
      email: emailVal,
      consent: true,
      page_url: location.href,
      ts: new Date().toISOString(),
    };

    // Try JSON POST if endpoint configured
    if (endpoint) {
      try {
        setStatus('Versturen…');
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          setStatus('Dank! We hebben uw e‑mail ontvangen.', 'ok');
          form.reset();
          return;
        }
        // fallthrough to mailto on non-2xx
      } catch (_) {
        // fallthrough to mailto
      }
    }

    // Fallback: open mailto with prefilled body
    if (mailTo) {
      const subject = encodeURIComponent('Nieuwe lead via website');
      const body = encodeURIComponent(
        `E‑mail: ${emailVal}\nConsent: ja\nPagina: ${location.href}\nTijd: ${payload.ts}`
      );
      const href = `mailto:${mailTo}?subject=${subject}&body=${body}`;
      setStatus('Uw e‑mailprogramma wordt geopend…', 'info');
      window.location.href = href;
      return;
    }

    // Last resort: instruct to configure
    setStatus('Kon niet versturen. Stel data‑attributes in op <body>: data-form-endpoint of data-mail-to.', 'err');
  });
})();
