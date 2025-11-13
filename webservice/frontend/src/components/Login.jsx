import React, { useEffect, useState } from 'react';
import { loginUser, setAuth } from '../api.js';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => { setError(''); }, [email, password]);

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!email || !password) {
      setError('Please enter email and password.');
      return;
    }
    try {
      setLoading(true);
      const user = await loginUser({ email, password });
      setAuth(user);
      window.location.hash = '#/';
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const containerStyle = { maxWidth: 420, margin: '48px auto', padding: 16, border: '1px solid #ddd', borderRadius: 8, fontFamily: 'system-ui, sans-serif' };
  const labelStyle = { display: 'block', marginBottom: 8 };
  const inputStyle = { width: '100%', padding: 8, marginTop: 4 };

  return (
    <div style={containerStyle}>
      <h2 style={{ marginTop: 0 }}>Sign in</h2>
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
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" style={inputStyle} required />
        </label>
        <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
          <button type="submit" disabled={loading}>{loading ? 'Signing in…' : 'Sign in'}</button>
          <a href="#/register">Create an account</a>
        </div>
      </form>
    </div>
  );
}
