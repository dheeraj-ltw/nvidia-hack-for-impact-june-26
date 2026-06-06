import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JARVIS",
  description:
    "JARVIS — a real-time legal & de-escalation copilot for frontline officers. Assistive, cited, audit-logged.",
  icons: { icon: "/jarvis-logo.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
