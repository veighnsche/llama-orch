// Created by: TEAM-DX-004
// List available variants for a specific story

use crate::error::{DxError, Result};
use crate::fetcher::Fetcher;
use colored::*;
use scraper::{Html, Selector};

/// List variants command handler
pub struct ListVariantsCommand {
    fetcher: Fetcher,
}

impl ListVariantsCommand {
    pub fn new() -> Self {
        Self {
            fetcher: Fetcher::new(),
        }
    }
    
    /// List all variants for a story
    pub async fn list_variants(&self, story_url: &str) -> Result<Vec<VariantInfo>> {
        let html = self.fetcher.fetch_page(story_url).await?;
        let variants = self.parse_variants(&html, story_url)?;
        
        // TEAM-DX-004: Handle edge case where story has only 1 variant
        // Histoire auto-redirects to variant-0, so we need to detect this
        if variants.is_empty() && !story_url.contains("variantId=") {
            // Try fetching with variant-0 appended
            let base_url = story_url.split('?').next().unwrap_or(story_url);
            let story_id = if let Some(id) = base_url.strip_prefix("http://localhost:6006/story/") {
                id.to_string()
            } else if let Some(id) = base_url.strip_prefix("http://localhost:6006/?path=/story/") {
                id.to_string()
            } else {
                return Ok(variants);
            };
            
            let variant_url = format!("{}?variantId={}-0", base_url, story_id);
            let variant_html = self.fetcher.fetch_page(&variant_url).await?;
            
            // Try to extract variant title from the page
            let document = Html::parse_document(&variant_html);
            let title_selector = Selector::parse("title").ok();
            let title = if let Some(sel) = title_selector {
                document.select(&sel)
                    .next()
                    .map(|el| el.text().collect::<String>())
                    .unwrap_or_else(|| "Default".to_string())
            } else {
                "Default".to_string()
            };
            
            // Return single variant
            return Ok(vec![VariantInfo {
                variant_id: format!("{}-0", story_id),
                title: title.split('|').next().unwrap_or("Default").trim().to_string(),
                url: variant_url,
            }]);
        }
        
        Ok(variants)
    }
    
    /// Parse variants from Histoire HTML
    fn parse_variants(&self, html: &str, story_url: &str) -> Result<Vec<VariantInfo>> {
        let document = Html::parse_document(html);
        
        // Histoire renders variant links in the sidebar
        // Look for elements with variant titles
        // The variant selector might be: button, a, or div with specific classes
        
        // Try multiple selectors to find variants
        let selectors = vec![
            "button[class*='variant']",
            "a[class*='variant']",
            "[role='button'][class*='variant']",
            "button",
            "a[href*='variantId']",
        ];
        
        let mut variants = Vec::new();
        let base_url = story_url.split('?').next().unwrap_or(story_url);
        
        // Extract story ID from URL for constructing variant URLs
        let story_id = if let Some(id) = base_url.strip_prefix("http://localhost:6006/story/") {
            id.to_string()
        } else if let Some(id) = base_url.strip_prefix("http://localhost:6006/?path=/story/") {
            id.to_string()
        } else {
            return Err(DxError::Parse("Could not extract story ID from URL".to_string()));
        };
        
        // Try to find variant links with variantId parameter
        let link_selector = Selector::parse("a[href*='variantId']")
            .map_err(|e| DxError::Parse(format!("Invalid selector: {}", e)))?;
        
        for element in document.select(&link_selector) {
            if let Some(href) = element.value().attr("href") {
                // Extract variant ID from href
                if let Some(variant_param) = href.split("variantId=").nth(1) {
                    let variant_id = variant_param.split('&').next().unwrap_or(variant_param);
                    
                    // Extract variant title from element text
                    let title = element.text().collect::<String>().trim().to_string();
                    
                    if !title.is_empty() {
                        let full_url = if href.starts_with("http") {
                            href.to_string()
                        } else {
                            format!("http://localhost:6006{}", href)
                        };
                        
                        variants.push(VariantInfo {
                            variant_id: variant_id.to_string(),
                            title,
                            url: full_url,
                        });
                    }
                }
            }
        }
        
        // If no variants found via links, try to construct them from story structure
        if variants.is_empty() {
            // Look for variant titles in the page
            let title_selectors = vec![
                "[class*='variant'] [class*='title']",
                "button[class*='htw']",
                "[role='button']",
            ];
            
            for selector_str in title_selectors {
                if let Ok(selector) = Selector::parse(selector_str) {
                    let mut index = 0;
                    for element in document.select(&selector) {
                        let title = element.text().collect::<String>().trim().to_string();
                        if !title.is_empty() && title.len() < 100 {
                            let variant_id = format!("{}-{}", story_id, index);
                            let url = format!("{}?variantId={}", base_url, variant_id);
                            
                            variants.push(VariantInfo {
                                variant_id: variant_id.clone(),
                                title,
                                url,
                            });
                            index += 1;
                        }
                    }
                    
                    if !variants.is_empty() {
                        break;
                    }
                }
            }
        }
        
        // Deduplicate by variant_id
        let mut seen = std::collections::HashSet::new();
        variants.retain(|v| seen.insert(v.variant_id.clone()));
        
        Ok(variants)
    }
    
    /// Print variants with URLs
    pub fn print_variants(&self, variants: &[VariantInfo], story_url: &str) {
        if variants.is_empty() {
            println!("{} No variants found for story", "✗".red());
            println!("  Story URL: {}", story_url.dimmed());
            println!();
            println!("  This could mean:");
            println!("    - The story has no variants defined");
            println!("    - The page structure is different than expected");
            println!("    - Histoire is still loading");
            return;
        }
        
        println!("{} Found {} variant{}\n", "✓".green(), variants.len(), if variants.len() == 1 { "" } else { "s" });
        
        // TEAM-DX-004: Special message for single-variant stories
        if variants.len() == 1 {
            println!("{}", "Note: This story has only 1 variant. Histoire auto-redirects to it.".dimmed());
            println!();
        }
        
        for (index, variant) in variants.iter().enumerate() {
            println!("{}. {}", index, variant.title.yellow().bold());
            println!("   {}", variant.url.cyan());
            println!();
        }
        
        println!("Total: {} variant{}", variants.len(), if variants.len() == 1 { "" } else { "s" });
    }
    
    /// Print variants in copy-pastable format
    pub fn print_variants_copy_pastable(&self, variants: &[VariantInfo]) {
        for variant in variants {
            println!("# {}", variant.title);
            println!("{}", variant.url);
            println!();
        }
    }
}

impl Default for ListVariantsCommand {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct VariantInfo {
    pub variant_id: String,
    pub title: String,
    pub url: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_variants_command_creation() {
        let cmd = ListVariantsCommand::new();
        assert!(true); // Just verify it compiles
    }
}
