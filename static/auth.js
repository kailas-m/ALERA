(function () {
	function qs(sel) { return document.querySelector(sel); }
	function on(el, ev, fn) { el && el.addEventListener(ev, fn); }
	function showMsg(el, text, isError) {
		if (!el) return;
		el.textContent = text || "";
		el.style.color = isError ? "#dc2626" : "#16a34a";
	}

	function ensureToastStyles() {
		if (document.getElementById('auth-toast-style')) return;
		var style = document.createElement('style');
		style.id = 'auth-toast-style';
		style.textContent = 
			".auth-toast{position:fixed;right:20px;bottom:20px;background:#111827;color:#fff;padding:12px 14px;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.2);z-index:99999;font-size:14px;}"+
			".auth-toast.success{background:#065f46}"+
			".auth-toast.error{background:#991b1b}";
		document.head.appendChild(style);
	}

	function toast(message, type) {
		ensureToastStyles();
		var el = document.createElement('div');
		el.className = 'auth-toast ' + (type === 'error' ? 'error' : 'success');
		el.textContent = message;
		document.body.appendChild(el);
		setTimeout(function(){ el.remove(); }, 2000);
	}

	// --- Local history management (stored in localStorage) ---
	function loadAuthHistory() {
		try {
			var raw = localStorage.getItem('alera_auth_history');
			return raw ? JSON.parse(raw) : [];
		} catch (e) { return []; }
	}
	function saveAuthHistoryEntry(entry) {
		try {
			var items = loadAuthHistory();
			items.unshift(entry);
			// keep recent 50
			if (items.length > 50) items = items.slice(0,50);
			localStorage.setItem('alera_auth_history', JSON.stringify(items));
		} catch (e) { console.error('Failed to save auth history', e); }
	}

	function formatDate(ts) {
		var d = new Date(ts);
		return d.toLocaleString();
	}

	// create person icon button and history dropdown
	function ensurePersonUI() {
		var container = document.querySelector('.auth-buttons');
		if (!container) return null;
		// hide existing signup/login buttons if present
		try {
			var signupBtn = container.querySelector('button.auth-button[onclick*="signup.html"]');
			var loginBtn = container.querySelector('button.auth-button[onclick*="login.html"]');
			if (signupBtn) signupBtn.style.display = 'none';
			if (loginBtn) loginBtn.style.display = 'none';
		} catch (e) {}
		var existing = container.querySelector('.person-menu');
		if (existing) return existing;
		var wrap = document.createElement('div');
		wrap.className = 'person-menu';
		wrap.style.position = 'relative';
		wrap.style.display = 'inline-block';

		var btn = document.createElement('button');
		btn.className = 'auth-button person-btn';
		btn.title = 'Account';
		// Prefer FontAwesome user icon if available, otherwise fallback to unicode glyph
		if (window.FontAwesome || document.querySelector('.fa') ) {
			btn.innerHTML = '<i class="fa fa-user" aria-hidden="true"></i>';
		} else {
			btn.innerHTML = '\u{1F464}';
		}
		btn.style.fontSize = '18px';
		btn.style.padding = '8px 10px';

		var panel = document.createElement('div');
		panel.className = 'person-panel';
		panel.style.position = 'absolute';
		panel.style.right = '0';
		panel.style.top = 'calc(100% + 8px)';
		panel.style.minWidth = '220px';
		panel.style.background = '#fff';
		panel.style.border = '1px solid rgba(0,0,0,.12)';
		panel.style.boxShadow = '0 8px 24px rgba(0,0,0,.12)';
		panel.style.borderRadius = '8px';
		panel.style.padding = '8px';
		panel.style.zIndex = 99999;
		panel.style.display = 'none';

	var title = document.createElement('div');
	title.style.fontWeight = '600';
	title.style.marginBottom = '6px';
	title.className = 'person-panel-title';
	title.textContent = 'Account History';
	panel.appendChild(title);

		var list = document.createElement('div');
		list.className = 'person-history';
		list.style.maxHeight = '220px';
		list.style.overflow = 'auto';
		list.style.fontSize = '13px';
		panel.appendChild(list);

		var hr = document.createElement('hr'); hr.style.margin = '8px 0'; panel.appendChild(hr);
		var logoutBtn = document.createElement('button');
		logoutBtn.className = 'auth-button logout-btn';
		logoutBtn.textContent = 'Logout';
		logoutBtn.style.width = '100%';
		panel.appendChild(logoutBtn);

		wrap.appendChild(btn);
		wrap.appendChild(panel);

		btn.addEventListener('click', function (e) {
			e.stopPropagation();
			panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
			refreshHistoryList(list);
		});
		document.addEventListener('click', function () { panel.style.display = 'none'; });

		logoutBtn.addEventListener('click', function () {
			// Clear local flag and update UI immediately without reloading
			try { localStorage.removeItem('alera_logged_in'); } catch (e) {}
			try {
				// remove person menu
				var personEl = document.querySelector('.person-menu'); if (personEl) personEl.remove();
				// restore signup/login buttons
				var container = document.querySelector('.auth-buttons');
				if (container) {
					var signupBtn = container.querySelector('button.auth-button[onclick*="signup.html"]');
					var loginBtn = container.querySelector('button.auth-button[onclick*="login.html"]');
					if (signupBtn) { signupBtn.style.display = ''; signupBtn.onclick = function(){ window.location.href='signup.html'; }; }
					if (loginBtn) { loginBtn.style.display = ''; loginBtn.textContent = 'Log In'; loginBtn.onclick = function(){ window.location.href='login.html'; }; }
				}
			} catch (e) { console.error('Error clearing UI on logout', e); }

			// Attempt to sign out via Firebase if available, but don't force a reload which can disturb page state (Analysis)
			var auth = hasMod() ? getModAuth() : getCompatAuth();
			if (auth) {
				try {
					if (hasMod()) {
						window.firebaseMod.signOut(auth).then(function(){ toast('Signed out', 'success'); }).catch(function(err){ console.warn('Sign out failed', err); toast('Signed out locally', 'success'); });
					} else {
						auth.signOut().then(function(){ toast('Signed out', 'success'); }).catch(function(err){ console.warn('Sign out failed', err); toast('Signed out locally', 'success'); });
					}
				} catch (e) { console.warn('Sign out exception', e); }
			} else {
				// No auth object available; just show feedback
				toast('Signed out locally', 'success');
			}
		});

		container.appendChild(wrap);
		return wrap;
	}

	// Helper: check Firebase currentUser synchronously if available
	function getCurrentUserSync() {
		try {
			var auth = hasMod() ? getModAuth() : getCompatAuth();
			if (!auth) return null;
			return auth.currentUser || null;
		} catch (e) { return null; }
	}

	// If the user is known to be logged in by local flag, show UI immediately and verify with Firebase
	document.addEventListener('DOMContentLoaded', function () {
		try {
			var flag = localStorage.getItem('alera_logged_in');
			if (flag === '1') {
				// show UI right away so navigation feels instant
				ensurePersonUI();

				// after a short delay, verify with Firebase; if no real user, clear stale flag and UI
				setTimeout(function () {
					try {
						var user = getCurrentUserSync();
						if (!user) {
							try { localStorage.removeItem('alera_logged_in'); } catch (e) {}
							var person = document.querySelector('.person-menu'); if (person) person.remove();
							var container = document.querySelector('.auth-buttons');
							if (container) {
								var signupBtn = container.querySelector('button.auth-button[onclick*="signup.html"]');
								var loginBtn = container.querySelector('button.auth-button[onclick*="login.html"]');
								if (signupBtn) { signupBtn.style.display = ''; signupBtn.onclick = function(){ window.location.href='signup.html'; }; }
								if (loginBtn) { loginBtn.style.display = ''; loginBtn.textContent = 'Log In'; loginBtn.onclick = function(){ window.location.href='login.html'; }; }
							}
						}
					} catch (e) { /* ignore verification errors */ }
				}, 700);
			}
		} catch (e) {}
	});

	// (early-hide behavior handled by immediate ensurePersonUI above)

// Sync across tabs and handle page show (bfcache/back-forward) so UI doesn't linger on pages
window.addEventListener('storage', function (e) {
	try {
		if (e.key === 'alera_logged_in') {
			var val = e.newValue;
			if (val !== '1') {
				// removed or changed -> hide person UI and restore auth buttons
				var person = document.querySelector('.person-menu'); if (person) person.remove();
				var container = document.querySelector('.auth-buttons');
				if (container) {
					var signupBtn = container.querySelector('button.auth-button[onclick*="signup.html"]');
					var loginBtn = container.querySelector('button.auth-button[onclick*="login.html"]');
					if (signupBtn) { signupBtn.style.display = ''; signupBtn.onclick = function(){ window.location.href='signup.html'; }; }
					if (loginBtn) { loginBtn.style.display = ''; loginBtn.textContent = 'Log In'; loginBtn.onclick = function(){ window.location.href='login.html'; }; }
				}
			} else {
				// value set to '1' in another tab -> ensure UI
				ensurePersonUI();
			}
		}
	} catch (err) { console.warn('storage sync error', err); }
});

window.addEventListener('pageshow', function () {
    try {
        var flag = localStorage.getItem('alera_logged_in');
        if (flag !== '1') {
            var person = document.querySelector('.person-menu'); if (person) person.remove();
            var container = document.querySelector('.auth-buttons');
            if (container) {
                var signupBtn = container.querySelector('button.auth-button[onclick*="signup.html"]');
                var loginBtn = container.querySelector('button.auth-button[onclick*="login.html"]');
                if (signupBtn) { signupBtn.style.display = ''; signupBtn.onclick = function(){ window.location.href='signup.html'; }; }
                if (loginBtn) { loginBtn.style.display = ''; loginBtn.textContent = 'Log In'; loginBtn.onclick = function(){ window.location.href='login.html'; }; }
            }
        } else {
            // value set to '1' in another tab -> ensure UI
            ensurePersonUI();
            // verify Firebase user after a short delay and clear stale flag if needed
            // NOTE: only clear the flag if Firebase is actually available (avoid false logout on pages without SDK)
            setTimeout(function () {
                try {
                    // If neither compat nor modular Firebase is present, skip clearing (likely page doesn't load Firebase)
                    if (!(window.firebase || window.firebaseMod)) {
                        // Defer verification: don't clear local flag when SDK missing
                        return;
                    }
                    var user = getCurrentUserSync();
                    if (!user) {
                        try { localStorage.removeItem('alera_logged_in'); } catch(e){}
                        var person = document.querySelector('.person-menu'); if (person) person.remove();
                        var container = document.querySelector('.auth-buttons');
                        if (container) {
                            var signupBtn = container.querySelector('button.auth-button[onclick*="signup.html"]');
                            var loginBtn = container.querySelector('button.auth-button[onclick*="login.html"]');
                            if (signupBtn) { signupBtn.style.display = ''; signupBtn.onclick = function(){ window.location.href='signup.html'; }; }
                            if (loginBtn) { loginBtn.style.display = ''; loginBtn.textContent = 'Log In'; loginBtn.onclick = function(){ window.location.href='login.html'; }; }
                        }
                    }
                } catch (e) { /* ignore verification errors */ }
            }, 2000); // increase delay to allow SDK to initialize
        }
    } catch (e) {}
});

	function refreshHistoryList(listEl) {
		if (!listEl) return;
		var items = loadAuthHistory();
		listEl.innerHTML = '';
		if (!items || items.length === 0) { listEl.textContent = 'No history yet.'; return; }
		items.slice(0,50).forEach(function (it) {
			var row = document.createElement('div');
			row.style.padding = '6px 0';
			row.style.borderBottom = '1px dashed rgba(0,0,0,.06)';
			row.innerHTML = '<strong>'+ (it.type==='signup' ? 'Signed up' : 'Logged in') +'</strong> — <span style="color:#555">'+ (it.email || 'unknown') +'</span><br><small style="color:#777">'+ formatDate(it.ts) +'</small>';
			listEl.appendChild(row);
		});
	}


	function getCompatAuth() { return window.getFirebaseAuth ? window.getFirebaseAuth() : null; }
	function getModAuth() { return window.firebaseMod && window.firebaseMod.auth ? window.firebaseMod.auth : null; }
	function hasMod() { return !!(window.firebaseMod && window.firebaseMod.createUserWithEmailAndPassword); }

	// Signup handler
	(function setupSignup() {
		var form = qs('#signupForm');
		if (!form) return;
		var msg = document.createElement('p');
		msg.style.marginTop = '10px';
		form.appendChild(msg);

			on(form, 'submit', function (e) {
			e.preventDefault();
			var auth = hasMod() ? getModAuth() : getCompatAuth();
			if (!auth) { console.error('No Firebase auth available'); return; }
			var email = qs('#email') && qs('#email').value.trim();
			var password = qs('#password') && qs('#password').value;
				if (!email || !password) { showMsg(msg, 'Email and password are required', true); return; }
				if (password.length < 6) { showMsg(msg, 'Password must be at least 6 characters', true); return; }
			showMsg(msg, 'Creating account...', false);
			if (hasMod()) {
					window.firebaseMod.createUserWithEmailAndPassword(auth, email, password)
						.then(function () { 
							showMsg(msg, 'Account created! Redirecting...', false); 
							toast('Account created', 'success'); 
							try { saveAuthHistoryEntry({ type: 'signup', email: email, ts: Date.now() }); } catch(e){}
							try { localStorage.setItem('alera_logged_in','1'); } catch(e){}
							ensurePersonUI();
							window.location.href='index.html';
						})
						.catch(function (err) { var m=(err&&err.code)||''; var friendly = m==='auth/email-already-in-use'?'Email already in use': m==='auth/invalid-email'?'Invalid email address': m==='auth/weak-password'?'Password too weak': (err&&err.message)||'Signup failed'; showMsg(msg, friendly, true); toast(friendly,'error'); });
			} else {
				auth.createUserWithEmailAndPassword(email, password)
						.then(function () { 
							showMsg(msg, 'Account created! Redirecting...', false); 
							toast('Account created', 'success'); 
							try { saveAuthHistoryEntry({ type: 'signup', email: email, ts: Date.now() }); } catch(e){}
							try { localStorage.setItem('alera_logged_in','1'); } catch(e){}
							ensurePersonUI();
							window.location.href='index.html';
						})
						.catch(function (err) { var m=(err&&err.code)||''; var friendly = m==='auth/email-already-in-use'?'Email already in use': m==='auth/invalid-email'?'Invalid email address': m==='auth/weak-password'?'Password too weak': (err&&err.message)||'Signup failed'; showMsg(msg, friendly, true); toast(friendly,'error'); });
			}
		});
	})();

	// Login handler
	(function setupLogin() {
		var form = qs('#loginForm');
		if (!form) return;
		var msg = document.createElement('p');
		msg.style.marginTop = '10px';
		form.appendChild(msg);

			on(form, 'submit', function (e) {
			e.preventDefault();
			var auth = hasMod() ? getModAuth() : getCompatAuth();
			if (!auth) { console.error('No Firebase auth available'); return; }
			var email = qs('#email') && qs('#email').value.trim();
			var password = qs('#password') && qs('#password').value;
			if (!email || !password) { showMsg(msg, 'Email and password are required', true); return; }
			showMsg(msg, 'Signing in...', false);
			if (hasMod()) {
					window.firebaseMod.signInWithEmailAndPassword(auth, email, password)
						.then(function () { 
							showMsg(msg, 'Logged in! Redirecting...', false); 
							toast('Logged in', 'success'); 
							try { saveAuthHistoryEntry({ type: 'login', email: email, ts: Date.now() }); } catch(e){}
							try { localStorage.setItem('alera_logged_in','1'); } catch(e){}
							ensurePersonUI();
							setTimeout(function(){ window.location.href='index.html'; }, 1200); 
						})
						.catch(function (err) { var m=(err&&err.code)||''; var friendly = m==='auth/invalid-credential'?'Invalid email or password': m==='auth/too-many-requests'?'Too many attempts, try later': (err&&err.message)||'Login failed'; showMsg(msg, friendly, true); toast(friendly,'error'); });
			} else {
				auth.signInWithEmailAndPassword(email, password)
						.then(function () { 
							showMsg(msg, 'Logged in! Redirecting...', false); 
							toast('Logged in', 'success'); 
							try { saveAuthHistoryEntry({ type: 'login', email: email, ts: Date.now() }); } catch(e){}
							try { localStorage.setItem('alera_logged_in','1'); } catch(e){}
							ensurePersonUI();
							setTimeout(function(){ window.location.href='index.html'; }, 1200); 
						})
						.catch(function (err) { var m=(err&&err.code)||''; var friendly = m==='auth/invalid-credential'?'Invalid email or password': m==='auth/too-many-requests'?'Too many attempts, try later': (err&&err.message)||'Login failed'; showMsg(msg, friendly, true); toast(friendly,'error'); });
			}
		});
	})();

	// Auth state UI example (optional): toggle buttons if present
	(function trackAuthState() {
		var auth = hasMod() ? getModAuth() : getCompatAuth();
		if (!auth) return;
		var listen = hasMod() ? window.firebaseMod.onAuthStateChanged : auth.onAuthStateChanged.bind(auth);
		listen(auth, function (user) {
			var loginBtn = document.querySelector('button.auth-button[onclick*="login.html"]');
			var signupBtn = document.querySelector('button.auth-button[onclick*="signup.html"]');
			if (user) {
				// ensure person UI is visible and updated
				var person = ensurePersonUI();
				if (person) {
					var panel = person.querySelector('.person-panel');
					var list = person.querySelector('.person-history');
					refreshHistoryList(list);
					// show user display name or email in title if available
					var title = person.querySelector('.person-panel-title');
					if (title) {
						try {
							var name = (user.displayName || user.email || '').toString();
							if (name) title.textContent = name;
						} catch (e) {}
					}
					// replace any login button behavior with opening person menu
					if (loginBtn) { loginBtn.textContent = 'Account'; loginBtn.onclick = function(){ panel.style.display = panel.style.display === 'none' ? 'block' : 'none'; refreshHistoryList(list); }; }
					if (signupBtn) signupBtn.style.display = 'none';
				}
			} else {
				if (loginBtn) { loginBtn.textContent = 'Log In'; loginBtn.onclick = function(){ window.location.href='login.html'; }; }
				if (signupBtn) { signupBtn.style.display = ''; signupBtn.onclick = function(){ window.location.href='signup.html'; }; }
				// hide person UI if present
				var person = document.querySelector('.person-menu'); if (person) person.remove();
			}
		});
	})();
})();


