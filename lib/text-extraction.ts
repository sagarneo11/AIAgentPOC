import pdf from "pdf-parse";
import * as cheerio from "cheerio";

export async function extractTextFromPDF(buffer: Buffer): Promise<string> {
  try {
    const data = await pdf(buffer);
    return data.text;
  } catch (error) {
    throw new Error(`Failed to extract text from PDF: ${error}`);
  }
}

export async function extractTextFromURL(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch URL: ${response.statusText}`);
    }
    const html = await response.text();
    const $ = cheerio.load(html);
    
    // Remove script and style elements
    $("script, style").remove();
    
    // Extract text from main content areas
    const text = $("body").text();
    
    // Clean up whitespace
    return text
      .replace(/\s+/g, " ")
      .replace(/\n\s*\n/g, "\n")
      .trim();
  } catch (error) {
    throw new Error(`Failed to extract text from URL: ${error}`);
  }
}
