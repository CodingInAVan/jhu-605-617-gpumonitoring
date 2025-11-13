import React, { useEffect, useState } from 'react';
import { registerUser, setAuth } from '../api.js';

export default function Register() {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null); // { id, email, name, apiKey }

  useEffect(() => {
    setError('');
  }, [email, name, password, confirm]);

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    if (!email || !name || !password) {
      setError('Please fill in email, name, and password.');
      return;
    }
    const emailRe = /.+@.+\..+/;
    if (!emailRe.test(email)) {
      setError('Please enter a valid email address.');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }
    try {
      setLoading(true);
      const data = await registerUser({ email, name, password });
      setAuth(data); // Save auth data to localStorage
      setResult(data);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const containerStyle = { maxWidth: 480, margin: '24px auto', padding: 16, border: '1px solid #ddd', borderRadius: 8, fontFamily: 'system-ui, sans-serif' };
  const labelStyle = { display: 'block', marginBottom: 8 };
  const inputStyle = { width: '100%', padding: 8, marginTop: 4 };

  if (result) {
    return (
      <div style={containerStyle}>
        <h2 style={{ marginTop: 0 }}>Registration Successful</h2>
        <p>Welcome, {result.name}! Your account has been created for {result.email}.</p>
        <div style={{ background: '#f4f6f8', border: '1px solid #ccd', padding: 12, borderRadius: 6, margin: '12px 0' }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Your API key:</div>
          <code style={{ display: 'block', wordBreak: 'break-all', fontSize: 14 }}>{result.apiKey}</code>
        </div>
        <p style={{ fontSize: 12, color: '#555' }}>Copy and store this API key securely. You can use it to authenticate API calls from agents posting metrics.</p>
        <div style={{ display: 'flex', gap: 8 }}>
          <a href="#/" style={{ textDecoration: 'none' }}>
            <button>Go to Dashboard</button>
          </a>
          <button onClick={() => { navigator.clipboard?.writeText(result.apiKey).catch(() => {}); }}>Copy API Key</button>
        </div>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      <h2 style={{ marginTop: 0 }}>Create an account</h2>
      {error && (
        <div style={{ color: 'white', background: '#c0392b', padding: 8, borderRadius: 4, marginBottom: 12 }}>
          {error}
        </div>
      )}
      <form onSubmit={onSubmit}>
        <label style={labelStyle}>
          Email
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" style={inputStyle} required />
        </label>
        <label style={labelStyle}>
          Name
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" style={inputStyle} required />
        </label>
        <label style={labelStyle}>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" style={inputStyle} required />
        </label>
        <label style={labelStyle}>
          Confirm Password
          <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} placeholder="••••••••" style={inputStyle} required />
        </label>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button type="submit" disabled={loading}>{loading ? 'Registering…' : 'Register'}</button>
          <a href="#/" style={{ textDecoration: 'none' }}>
            <button type="button">Cancel</button>
          </a>
        </div>
      </form>
    </div>
  );
}
