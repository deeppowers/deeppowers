#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <unicode/unistr.h>

namespace deeppowers {

class TextProcessor {
public:
    TextProcessor();
    ~TextProcessor();

    // Configuration
    void set_lower_case(bool value) { do_lower_case_ = value; }
    void set_strip_accents(bool value) { strip_accents_ = value; }
    void set_handle_chinese_chars(bool value) { handle_chinese_chars_ = value; }
    void set_handle_numbers(bool value) { handle_numbers_ = value; }
    void set_handle_urls(bool value) { handle_urls_ = value; }
    void set_handle_emails(bool value) { handle_emails_ = value; }
    void set_normalize_unicode(bool value) { normalize_unicode_ = value; }

    // Process text with all enabled rules
    std::string process(const std::string& text) const;

    // Individual processing steps
    std::string to_lower_case(const std::string& text) const;
    std::string strip_accents(const std::string& text) const;
    std::string handle_chinese_chars(const std::string& text) const;
    std::string normalize_numbers(const std::string& text) const;
    std::string normalize_urls(const std::string& text) const;
    std::string normalize_emails(const std::string& text) const;
    std::string normalize_unicode(const std::string& text) const;
    std::string clean_text(const std::string& text) const;

private:
    // Helper methods
    bool is_chinese_char(char32_t cp) const;
    bool is_whitespace(char32_t cp) const;
    bool is_control(char32_t cp) const;
    bool is_punctuation(char32_t cp) const;
    
    // Regular expressions for normalization
    std::regex url_pattern_;
    std::regex email_pattern_;
    std::regex number_pattern_;
    
    // Configuration flags
    bool do_lower_case_;
    bool strip_accents_;
    bool handle_chinese_chars_;
    bool handle_numbers_;
    bool handle_urls_;
    bool handle_emails_;
    bool normalize_unicode_;
};

} // namespace deeppowers 