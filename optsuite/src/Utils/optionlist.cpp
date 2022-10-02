/*
 * ==========================================================================
 *
 *       Filename:  optionlist.cpp
 *
 *    Description: 
 *
 *        Version:  1.0
 *        Created:  11/20/2020 02:46:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Haoyang Liu (@liuhy), liuhaoyang@pku.edu.cn
 *   Organization:  BICMR, PKU
 *
 * ==========================================================================
 */

#include <sstream>
#include <cstdio>
#include "OptSuite/Utils/optionlist.h"
#include "OptSuite/Utils/str_format.h"

namespace OptSuite { namespace Utils {
    OptionList::OptionList(const std::vector<RegOption>& op, std::shared_ptr<Logger> ptr){
        for (const auto& i : op){
            this->reg_options.insert({i.tag, i});
        }
        logger_ptr = std::move(ptr);
    }

    void OptionList::register_option(const RegOption& reg_opt){
        this->reg_options.insert({reg_opt.tag, reg_opt});
    }

    template<typename T>
    bool OptionList::check_option_t(const std::string& tag, const T& value, const OptionType input_type, const bool self_check) const {
        // get the option type
        const RegOption* prr = get_option(tag);

        if (!prr) return false;

        const RegOption &rr = *prr;

        // return false if wrong type
        if (rr.type != input_type){
            if (logger_ptr){
                logger_ptr->log_info("Option \'", tag, "\' should be ", 
                        to_string(rr.type), " type, got ", to_string(input_type), "\n");
            }
            return false;
        }

        // return false if failed in self-checking
        if (self_check){
            if (!(*rr.checker_ptr)(value)){
                if (logger_ptr){
                    logger_ptr->log_info("Failed to set option \'", tag, "\': ", rr.description, "\n");
                }
                return false;
            }
        }

        // check passed
        return true;
    }

    bool OptionList::set_string(const std::string& tag, const std::string& v){
        bool result = check_option_t(tag, v, OptionType::String);
        OPTSUITE_ASSERT(result);
        if (result) {
            options[tag].s = v;
            if (logger_ptr)
                logger_ptr->log_debug(
                        "Option \'", tag, "\' is set to ", options[tag].s, "\n");
        }
        return result;
    }

    bool OptionList::set_integer(const std::string& tag, const Index& v){
        bool result = check_option_t(tag, v, OptionType::Integer);
        OPTSUITE_ASSERT(result);
        if (result) {
            options[tag].i = v;
            if (logger_ptr)
                logger_ptr->log_debug(
                        "Option \'", tag, "\' is set to ", options[tag].i, "\n");
        }
        return result;
    }

    bool OptionList::set_scalar(const std::string& tag, const Scalar& v){
        bool result = check_option_t(tag, v, OptionType::Scalar);
        OPTSUITE_ASSERT(result);
        if (result) {
            options[tag].d = v;
            if (logger_ptr)
                logger_ptr->log_debug(
                        "Option \'", tag, "\' is set to ", options[tag].d, "\n");
        }
        return result;
    }

    bool OptionList::set_bool(const std::string& tag, const bool& v){
        bool result = check_option_t(tag, v, OptionType::Bool);
        OPTSUITE_ASSERT(result);
        if (result) {
            options[tag].b = v;
            if (logger_ptr)
                 logger_ptr->log_debug(
                         "Option \'", tag, "\' is set to ", std::boolalpha,
                          options[tag].b, "\n");
        }
        return result;
    }

#define OPTSUITE_OPTION_LIST_GETTER(funname, T, OT, tok, vv) \
    T OptionList::get_ ## funname(const std::string& tag) const { \
        bool result = check_option_t(tag, T(vv), OptionType::OT, false); \
        OPTSUITE_ASSERT(result); \
        if (result){ \
            auto it = options.find(tag); \
            if (it != options.cend()) \
                return it->second.tok; \
            else \
                return reg_options.find(tag)->second.default_value.tok; \
        } \
        return vv; \
    }

    OPTSUITE_OPTION_LIST_GETTER(string, std::string, String, s, "")
    OPTSUITE_OPTION_LIST_GETTER(integer, Index, Integer, i, 0)
    OPTSUITE_OPTION_LIST_GETTER(scalar, Scalar, Scalar, d, 0)
    OPTSUITE_OPTION_LIST_GETTER(bool, bool, Bool, b, false)

#undef OPTSUITE_OPTION_LIST_GETTER

    std::string OptionList::to_string(OptionType t){
        switch(t){
            case OptionType::Bool:
                return "boolean";
            case OptionType::Integer:
                return "integer";
            case OptionType::Scalar:
                return "scalar";
            case OptionType::String:
                return "string";
            default:
                return "unknown";
        }
    }

    void OptionList::set_from_file(const std::string& filename){
        std::ifstream infile(filename);
        std::string buffer, tag, value;
        int line_number = 0;

        if (!infile.is_open()){
            if (logger_ptr)
                logger_ptr->log_info("Failed to open config file \"", filename, "\".\n");
            return;
        }

        // read line by line
        while (std::getline(infile, buffer)){
            ++line_number;

            // ignore comment line and empty line
            if (buffer.empty() || buffer[0] == '#')
                continue;

            // find option tag
            size_t pos = buffer.find_first_of(" \t");
            if (pos == std::string::npos || pos == 0) {
                if (logger_ptr)
                    logger_ptr->log_format("Line %d is illegal, skipped.\n", line_number);
                continue;
            }
            tag = buffer.substr(0, pos);
            while (std::isspace(buffer[pos++]));
            value = buffer.substr(pos - 1);

            // 'tag' and 'value' should have desired value
            set_from_string(tag, value);
        }

        infile.close();
    }

    void OptionList::set_from_string(const std::string& tag, const std::string& value){
        // obtain option type
        const RegOption* prr = get_option(tag);
        // option not registered
        if (!prr) return;

        try {
            switch (prr->type){
                case OptionType::String:
                    set_string(tag, value);
                    break;
                case OptionType::Integer:
                    set_integer(tag, std::stol(value));
                    break;
                case OptionType::Scalar:
                    set_scalar(tag, std::stod(value));
                    break;
                case OptionType::Bool:
                    // only accept true or false (case sensitive)
                    if (value == "true")
                        set_bool(tag, true);
                    else if (value == "false")
                        set_bool(tag, false);
                    else
                        throw std::invalid_argument("");

                    bool bvalue;
                    std::istringstream(value) >> std::boolalpha >> bvalue;
                    set_bool(tag, bvalue);
                    break;
                default:
                    break;
            }
        }
        catch (std::invalid_argument&) {
            if (logger_ptr)
                logger_ptr->log_format("Invalid argument \'%s\' for option \'%s\'.\n",
                        value.c_str(), tag.c_str());
        }
    }

    void OptionList::set_from_cmd_line(int argc ,char *argv[], const std::string& custom_prefix){
        const char *prefix =
            custom_prefix.empty() ? OPTSUITE_OPTION_CMD_PREFIX : custom_prefix.c_str();
        std::string key_value, tag, value;
        std::string missing_arg_s = str_format(
                "Error: missing argument after the %s option\n", prefix);
        std::string wrong_arg_s = str_format(
                "Error: tag=value should follow the %s option\n", prefix);

        for (int i = 1; i < argc; ++i){
            std::string arg = argv[i];
            if (arg == prefix){
                if (i < argc - 1){
                    key_value = argv[++i];
                } else {
                    if (logger_ptr)
                        logger_ptr->log_info(missing_arg_s);
                    return;
                }
                // key_value should have format "tag=value"
                // split tag and value
                size_t pos = key_value.find_first_of('=');
                if (pos == std::string::npos || pos == 0) {
                    if (logger_ptr)
                        logger_ptr->log_info(wrong_arg_s);
                    continue;
                }
                tag = key_value.substr(0, pos);
                value = key_value.substr(pos + 1);

                // call set_from_string
                set_from_string(tag, value);

            }
        }
    }

    const RegOption* OptionList::get_option(const std::string& tag) const {
        auto reg_it = reg_options.find(tag);

        if (reg_it == reg_options.cend()){
            // return nullptr if not found
            if (logger_ptr){
                logger_ptr->log_info("Option \'", tag, "\' is not registered.\n");
            }
            return nullptr;
        } else {
            return &reg_it->second;
        }
    }

    void OptionList::show(std::ostream& out, bool show_des) const {
        out << "All Registered Options:" << std::endl;
        out << "------------------------------------------------" << std::endl;

        // determine max length of tag and value
        size_t tag_len = 0, def_value_len = 0, value_len = 0;
        std::string value_s, d_value_s;
        std::vector<std::tuple<std::string, std::string, std::string>> key_value_s;
        key_value_s.reserve(reg_options.size());
        for (const auto & reg_option : reg_options){
            if (reg_option.first.size() > tag_len)
                tag_len = reg_option.first.size();

            d_value_s = reg_option.second.to_string_v();

            if (d_value_s.size() > def_value_len)
                def_value_len = d_value_s.size();

            // check options that is already set
            value_s = to_string_v(reg_option.first);

            if (value_s.size() > value_len)
                value_len = value_s.size();

            // record the <key, default_value, value> tuple
            key_value_s.emplace_back(reg_option.first, d_value_s, value_s);
        }

        tag_len = std::max((size_t)4, tag_len);
        def_value_len = std::max((size_t)7, def_value_len);
        value_len = std::max((size_t)7, value_len);

        // generate format string buffer
        const char *format = "%*s  %8s  %*s  %*s  %s\n";
        const char *des_str = show_des ? "description" : "";

        // print out
        out << cstr_format(format,
                tag_len, "name", "type", def_value_len, "default",
                value_len, "value", des_str).get();

        for (auto& i : key_value_s){
            const RegOption&rr = reg_options.at(std::get<0>(i));
            out << cstr_format(format,
                    tag_len,       std::get<0>(i).c_str(),
                                   OptionList::to_string(rr.type).c_str(), 
                    def_value_len, std::get<1>(i).c_str(),
                    value_len,     std::get<2>(i).c_str(),
                                   show_des ? rr.description.c_str() : ""
                                   ).get();

        }
    }

    void OptionList::show(bool show_des) const {
        show(logger_ptr->get_stream(), show_des);
    }

    std::ostream& operator<<(std::ostream& out, const OptionList& ol){
        ol.show(out, false);
        return out;
    }

    std::string OptionList::to_string_v(const std::string& tag) const {
        if (!is_initialized(tag))
            return "(default)";
        return RegOption::to_string_v(reg_options.at(tag).type, options.at(tag));
    }

    RegOption::RegOption(){
        tag = "";
        type = OptionType::Unknown;
        default_value = {"", 0_s, 0_i, false};
        checker_ptr = std::make_shared<OptionChecker>();
    }

    RegOption::RegOption(const std::string& tag, const std::string& v, 
            const std::string& des, OptionChecker_ptr ptr){
        this->default_value.s = v;
        this->type = OptionType::String;
        this->tag = tag;
        this->checker_ptr = std::move(ptr);
        this->description = des;
    }

    RegOption::RegOption(const std::string& tag, const Index& v,
            const std::string& des, OptionChecker_ptr ptr){
        this->default_value.i = v;
        this->type = OptionType::Integer;
        this->tag = tag;
        this->checker_ptr = std::move(ptr);
        this->description = des;
    }

    RegOption::RegOption(const std::string& tag, const Scalar& v,
            const std::string& des, OptionChecker_ptr ptr){
        this->default_value.d = v;
        this->type = OptionType::Scalar;
        this->tag = tag;
        this->checker_ptr = std::move(ptr);
        this->description = des;
    }

    RegOption::RegOption(const std::string& tag, const bool& v,
            const std::string& des, OptionChecker_ptr ptr){
        this->default_value.b = v;
        this->type = OptionType::Bool;
        this->tag = tag;
        this->checker_ptr = std::move(ptr);
        this->description = des;
    }

    RegOption::RegOption(const std::string& tag, const char *v,
            const std::string& des, OptionChecker_ptr ptr) : 
        RegOption(tag, std::string(v), des, std::move(ptr)) {}

    std::string RegOption::to_string_v(OptionType type, const OptionValue& value){
        std::string value_s;
        switch (type) {
            case OptionType::String:
                value_s = value.s.empty() ? "(empty)" : value.s;
                break;
            case OptionType::Integer:
                value_s = std::to_string(value.i);
                break;
            case OptionType::Scalar:
                value_s = str_format("%g", value.d);
                break;
            case OptionType::Bool:
                value_s = value.b ? "true" : "false";
                break;
            default:
                value_s = "";
                break;
        }
        return value_s;
    }

    std::string RegOption::to_string_v() const {
        return to_string_v(type, default_value);
    }
}}
