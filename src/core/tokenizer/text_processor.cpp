#include "text_processor.hpp"
#include <unicode/uchar.h>
#include <unicode/normalizer2.h>
#include <algorithm>

namespace deeppowers {

TextProcessor::TextProcessor()
    : url_pattern_(R"((https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}))")
    , email_pattern_(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
    , number_pattern_(R"(\d+(?:\.\d+)?)")
    , do_lower_case_(true)
    , strip_accents_(true)
    , handle_chinese_chars_(true)
    , handle_numbers_(true)
    , handle_urls_(true)
    , handle_emails_(true)
    , normalize_unicode_(true) {
}

TextProcessor::~TextProcessor() = default;

std::string TextProcessor::process(const std::string& text) const {
    std::string processed = text;
    
    // Apply all enabled processing steps
    if (normalize_unicode_) {
        processed = normalize_unicode(processed);
    }
    if (do_lower_case_) {
        processed = to_lower_case(processed);
    }
    if (strip_accents_) {
        processed = strip_accents(processed);
    }
    if (handle_chinese_chars_) {
        processed = handle_chinese_chars(processed);
    }
    if (handle_numbers_) {
        processed = normalize_numbers(processed);
    }
    if (handle_urls_) {
        processed = normalize_urls(processed);
    }
    if (handle_emails_) {
        processed = normalize_emails(processed);
    }
    
    return clean_text(processed);
}

std::string TextProcessor::to_lower_case(const std::string& text) const {
    icu::UnicodeString ustr(text.c_str());
    ustr.toLower();
    std::string result;
    ustr.toUTF8String(result);
    return result;
}

std::string TextProcessor::strip_accents(const std::string& text) const {
    UErrorCode status = U_ZERO_ERROR;
    const icu::Normalizer2* normalizer = icu::Normalizer2::getNFKDInstance(status);
    if (U_FAILURE(status)) {
        return text;
    }
    
    icu::UnicodeString ustr(text.c_str());
    icu::UnicodeString normalized = normalizer->normalize(ustr, status);
    if (U_FAILURE(status)) {
        return text;
    }
    
    // Remove diacritical marks
    icu::UnicodeString result;
    for (int32_t i = 0; i < normalized.length(); i++) {
        UChar32 ch = normalized.char32At(i);
        if (u_charType(ch) != U_NON_SPACING_MARK) {
            result.append(ch);
        }
    }
    
    std::string output;
    result.toUTF8String(output);
    return output;
}

std::string TextProcessor::handle_chinese_chars(const std::string& text) const {
    icu::UnicodeString ustr(text.c_str());
    icu::UnicodeString result;
    
    for (int32_t i = 0; i < ustr.length(); i++) {
        UChar32 ch = ustr.char32At(i);
        result.append(ch);
        if (is_chinese_char(ch)) {
            result.append(' ');
        }
    }
    
    std::string output;
    result.toUTF8String(output);
    return output;
}

std::string TextProcessor::normalize_numbers(const std::string& text) const {
    return std::regex_replace(text, number_pattern_, " <NUMBER> ");
}

std::string TextProcessor::normalize_urls(const std::string& text) const {
    return std::regex_replace(text, url_pattern_, " <URL> ");
}

std::string TextProcessor::normalize_emails(const std::string& text) const {
    return std::regex_replace(text, email_pattern_, " <EMAIL> ");
}

std::string TextProcessor::normalize_unicode(const std::string& text) const {
    UErrorCode status = U_ZERO_ERROR;
    const icu::Normalizer2* normalizer = icu::Normalizer2::getNFKCInstance(status);
    if (U_FAILURE(status)) {
        return text;
    }
    
    icu::UnicodeString ustr(text.c_str());
    icu::UnicodeString normalized = normalizer->normalize(ustr, status);
    if (U_FAILURE(status)) {
        return text;
    }
    
    std::string output;
    normalized.toUTF8String(output);
    return output;
}

std::string TextProcessor::clean_text(const std::string& text) const {
    icu::UnicodeString ustr(text.c_str());
    icu::UnicodeString result;
    bool prev_is_whitespace = true;
    
    for (int32_t i = 0; i < ustr.length(); i++) {
        UChar32 ch = ustr.char32At(i);
        
        // Skip control characters
        if (is_control(ch)) {
            continue;
        }
        
        // Replace whitespace with single space
        if (is_whitespace(ch)) {
            if (!prev_is_whitespace) {
                result.append(' ');
                prev_is_whitespace = true;
            }
            continue;
        }
        
        // Add space around punctuation
        if (is_punctuation(ch)) {
            if (!prev_is_whitespace) {
                result.append(' ');
            }
            result.append(ch);
            result.append(' ');
            prev_is_whitespace = true;
            continue;
        }
        
        result.append(ch);
        prev_is_whitespace = false;
    }
    
    std::string output;
    result.toUTF8String(output);
    
    // Trim leading/trailing whitespace
    output.erase(0, output.find_first_not_of(" "));
    output.erase(output.find_last_not_of(" ") + 1);
    
    return output;
}

bool TextProcessor::is_chinese_char(char32_t cp) const {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool TextProcessor::is_whitespace(char32_t cp) const {
    return u_isspace(cp) || cp == 0x200B;  // Include zero-width space
}

bool TextProcessor::is_control(char32_t cp) const {
    return (cp >= 0x00 && cp <= 0x1F) ||
           (cp >= 0x7F && cp <= 0x9F) ||
           cp == 0x200B;  // zero-width space
}

bool TextProcessor::is_punctuation(char32_t cp) const {
    return u_ispunct(cp);
}

} // namespace deeppowers 