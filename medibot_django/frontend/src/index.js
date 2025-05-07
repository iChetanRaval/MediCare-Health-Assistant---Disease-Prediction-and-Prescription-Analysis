import React from 'react';
import ReactDOM from 'react-dom';
import { ClerkProvider } from '@clerk/clerk-react';
import App from './App';

const clerkFrontendAPI = "https://ethical-termite-7.clerk.accounts.dev"; // Replace with Clerk Frontend API

ReactDOM.render(
  <ClerkProvider frontendApi={clerkFrontendAPI}>
    <React.StrictMode>
      <App />
    </React.StrictMode>
  </ClerkProvider>,
  document.getElementById('root')
);

