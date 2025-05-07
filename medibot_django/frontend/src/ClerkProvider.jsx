import { ClerkProvider } from '@clerk/clerk-react';

export default function Provider({ children }) {
  const clerkKey = "sk_test_sAKNsiEv8PpooxBXuo7aPgHanaf9NYshwJ1xdfGfeY";

  return (
    <ClerkProvider frontendApi={clerkKey}>
      {children}
    </ClerkProvider>
  );
}
