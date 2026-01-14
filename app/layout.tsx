import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Knowledge Agent",
  description: "Chat with your documents using RAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
