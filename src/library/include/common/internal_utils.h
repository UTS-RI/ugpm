/**
 *  Author: Cedric LE GENTIL 
 *
 *  Copyright 2021 Cedric LE GENTIL
 *
 *  For any further question, recommendation or contribution
 *  le.gentil.cedric@gmail.com
 **/

#ifndef COMMON_INTERNAL_UTILS_H
#define COMMON_INTERNAL_UTILS_H

#include "common/math_utils.h"
#include "yaml-cpp/yaml.h"
#include <utility>
#include <numeric>

namespace celib
{
    

    // Function to sort indexes
    template <typename T>
    std::vector<int> SortIndexes(const std::vector<T> &v) {

        // initialize original index locations
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0); 

        // sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2) {return v[i1] < v[i2];});

        return idx;
    }


    // Tool to sort vector of vector and keep track of original indexes
    template <class T>
    class SortIndexTracker2
    {

        public:
            SortIndexTracker2( std::vector<std::vector<T> >& data):
                data_(&data)
            {
                std::vector<T> temp_data;
                std::vector<std::pair<int, int> > temp_map;
                for(int i = 0; i < data_->size(); ++i)
                {
                    temp_data.insert(temp_data.end(), data_->at(i).begin(), data_->at(i).end());
                    for(int j = 0; j < data_->at(i).size(); ++j)
                    {
                        temp_map.push_back(std::make_pair(i,j));
                    }
                }
                
                std::vector<int> sorted_indexes = SortIndexes(temp_data);

                for(int i = 0; i < sorted_indexes.size(); ++i)
                {
                    index_map_.push_back(temp_map[sorted_indexes[i]]);
                }
            }

            T get(const int i) const
            {
                return (*data_)[index_map_[i].first][index_map_[i].second];
            }

            std::pair<int, int> getIndexPair(const int i) const
            {
                return index_map_[i];
            }

            std::vector<int> getIndexVector(const int vec_i) const
            {
                std::vector<int> output;
                for(int i = 0; i < index_map_.size(); ++i)
                {
                    if(index_map_[i].first == vec_i) output.push_back(i);
                }
                return output;
            }

            template <class U>
            std::vector<U> getVector(const std::vector<U>& data, const int vec_i) const
            {
                std::vector<U> output;
                for(int i = 0; i < index_map_.size(); ++i)
                {
                    if(index_map_[i].first == vec_i) output.push_back(data[i]);
                }
                return output;
            }


            // Get the 1D index corresponding to the 2D indices
            int getIndex(const int i, const int j) const
            {
                for(int k = 0; k < index_map_.size(); ++k)
                {
                    if((index_map_[k].first == i) && (index_map_[k].second == j))
                    {
                        return k;
                    }
                }
                return -1;
            }

            // Get from the 2D indices in a 1D structure
            template <class U>
            U get(const int i, const int j, const std::vector<U>& data) const
            {
                for(int k = 0; k < index_map_.size(); ++k)
                {
                    if((index_map_[k].first == i) && (index_map_[k].second == j))
                    {
                        return data[k];
                    }
                }
                throw std::range_error("SortIndexTracker2: trying to 'get' with indices not present in the index_map");
            }


            // Get from the 2D indices in a 1D structure
            template <class U>
            std::vector<U> applySort(const std::vector<std::vector<U> >& data) const
            {
                std::vector<U> output;
                for(int k = 0; k < index_map_.size(); ++k)
                {
                    output.push_back(data[index_map_[k].first][index_map_[k].second]);
                }
                return output;
            }


            T back() const
            {
                return (*data_)[index_map_.back().first][index_map_.back().second];
            }

            int size() const
            {
                return index_map_.size();
            }
            
            T getSmallestGap()
            {
                T diff = get(1) - get(0);
                for(int i = 1; i < (size()-1); ++i)
                {
                    diff = get(i+1) - get(i);
                }
                return diff;
            }

        private:
            std::vector<std::vector<T> >* data_;

            std::vector<std::pair<int, int> > index_map_;
            

    };




}


#endif
