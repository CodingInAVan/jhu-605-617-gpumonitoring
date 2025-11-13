import React, { useEffect, useMemo, useState } from 'react';
import App from './App.jsx';
import Register from './components/Register.jsx';
import Login from './components/Login.jsx';
import { getAuth } from './api.js';

function useHashRoute() {
  const [hash, setHash] = useState(() => window.location.hash || '#/');
  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash || '#/');
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
  return hash;
}

export default function Root() {
  const hash = useHashRoute();
  const authed = useMemo(() => !!getAuth()?.apiKey, [hash]);

  // Guard: if not authenticated, redirect to /login except for /login and /register
  useEffect(() => {
    const isLogin = hash.startsWith('#/login');
    const isRegister = hash.startsWith('#/register');
    if (!authed && !isLogin && !isRegister) {
      window.location.hash = '#/login';
    }
  }, [hash, authed]);

  if (hash.startsWith('#/register')) return <Register />;
  if (hash.startsWith('#/login')) return <Login />;
  return <App />;
}
