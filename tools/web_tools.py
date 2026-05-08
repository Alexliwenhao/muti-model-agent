"""Web tools for Claude Code style agent

Claude Code web tools:
- WebFetch: Fetch content from URLs
- WebSearch: Search the web
- Support for markdown conversion
"""
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time

from langchain_core.tools import BaseTool
from pydantic import BaseModel


class WebFetchInput(BaseModel):
    """Input schema for WebFetch tool"""
    url: str = {"description": "URL to fetch"}


class WebFetchTool(BaseTool):
    """Fetch content from a URL"""
    name = "WebFetch"
    description = "Fetch content from a URL. Use this to read web pages, documentation, etc."
    args_schema = WebFetchInput

    def _run(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; Claude Code/1.0; +https://claude.ai)"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            
            if "text/html" in content_type:
                return self._parse_html(response.text, url)
            elif "application/json" in content_type:
                return response.text[:5000]
            else:
                return response.text[:5000]

        except requests.exceptions.RequestException as e:
            return f"Error fetching {url}: {str(e)}"

    def _parse_html(self, html: str, url: str) -> str:
        """Parse HTML and extract readable content"""
        soup = BeautifulSoup(html, 'html.parser')
        
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()

        title = soup.title.string if soup.title else "No Title"
        
        main_content = soup.find("main") or soup.find("article") or soup.find("body")
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)

        result = f"""URL: {url}
Title: {title}
Content:

{clean_text[:5000]}"""

        return result


class WebSearchInput(BaseModel):
    """Input schema for WebSearch tool"""
    query: str = {"description": "Search query"}
    num_results: Optional[int] = {"description": "Number of results to return", "default": 5}


class WebSearchTool(BaseTool):
    """Search the web"""
    name = "WebSearch"
    description = "Search the web for information. Use this to find documentation, tutorials, or current information."
    args_schema = WebSearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        try:
            results = self._execute_search(query, num_results)
            return self._format_results(results)
        except Exception as e:
            return f"Search error: {str(e)}"

    def _execute_search(self, query: str, num_results: int) -> List[Dict[str, str]]:
        """Execute web search"""
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Claude Code/1.0)"
        }
        
        search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        for result in soup.find_all("div", class_="result")[:num_results]:
            title_elem = result.find("a", class_="result__a")
            link_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")

            if title_elem and link_elem:
                results.append({
                    "title": title_elem.text.strip(),
                    "url": link_elem.get("href", ""),
                    "snippet": snippet_elem.text.strip() if snippet_elem else ""
                })

        return results

    def _format_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results"""
        if not results:
            return "No results found."

        output = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. [{result['title']}]({result['url']})\n"
            if result.get('snippet'):
                output += f"   {result['snippet']}\n"
            output += "\n"

        return output


class URLInfoInput(BaseModel):
    """Input schema for URLInfo tool"""
    url: str = {"description": "URL to analyze"}


class URLInfoTool(BaseTool):
    """Get information about a URL"""
    name = "URLInfo"
    description = "Get information about a URL - domain, protocol, path, etc."
    args_schema = URLInfoInput

    def _run(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return f"""URL Analysis:
Protocol: {parsed.scheme}
Domain: {parsed.netloc}
Path: {parsed.path}
Query: {parsed.query}
Fragment: {parsed.fragment}
"""
        except Exception as e:
            return f"Error: {str(e)}"


def get_web_tools() -> List[BaseTool]:
    """Get all web tools"""
    return [
        WebFetchTool(),
        WebSearchTool(),
        URLInfoTool()
    ]
