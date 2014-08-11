#ifndef OSMIUM_TAGS_KEY_FILTER_HPP
#define OSMIUM_TAGS_KEY_FILTER_HPP

/*

Copyright 2012 Jochen Topf <jochen@topf.org> and others (see README).

This file is part of Osmium (https://github.com/joto/osmium).

Osmium is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License or (at your option) the GNU
General Public License as published by the Free Software Foundation, either
version 3 of the Licenses, or (at your option) any later version.

Osmium is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License and the GNU
General Public License for more details.

You should have received a copy of the Licenses along with Osmium. If not, see
<http://www.gnu.org/licenses/>.

*/

#include <functional>
#include <vector>
#include <boost/foreach.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <osmium/osm/tag.hpp>
#include <osmium/osm/tag_list.hpp>

namespace Osmium {

    namespace Tags {

        class KeyFilter : public std::unary_function<const Osmium::OSM::Tag&, bool> {

            struct rule_t {
                bool result;
                std::string key;

                rule_t(bool r, const char* k) :
                    result(r),
                    key(k) {
                }

            };

            std::vector<rule_t> m_rules;
            bool m_default_result;

        public:

            typedef boost::filter_iterator<KeyFilter, Osmium::OSM::TagList::const_iterator> iterator;

            KeyFilter(bool default_result) :
                m_rules(),
                m_default_result(default_result) {
            }

            KeyFilter& add(bool result, const char* key) {
                m_rules.push_back(rule_t(result, key));
                return *this;
            }

            bool operator()(const Osmium::OSM::Tag& tag) const {
                BOOST_FOREACH(const rule_t& rule, m_rules) {
                    if (tag.key() == rule.key) {
                        return rule.result;
                    }
                }
                return m_default_result;
            }

        };

    } // namespace Tags

} // namespace Osmium

#endif // OSMIUM_TAGS_KEY_FILTER_HPP
