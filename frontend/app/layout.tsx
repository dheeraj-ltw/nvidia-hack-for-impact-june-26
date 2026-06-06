import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Patrol Assist",
  description: "AI decision-support for police patrol — assistive, cited, audit-logged.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
