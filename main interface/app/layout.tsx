import type React from "react";
import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Analytics } from "@vercel/analytics/next";
import { Suspense } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agri-Sage - Smart Farming Solutions",
  description:
    "AI-powered agricultural advisory platform for crop disease identification, weather insights, and market trends",
  generator: "v0.app",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        {/* Page container wrapper */}
        <div className="max-w-5xl mx-auto px-4">
          <Suspense fallback={null}>{children}</Suspense>
          <Analytics />
        </div>
      </body>
    </html>
  );
}
