#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>

namespace nanodb {

    class MMapHandler {
    public:
        MMapHandler();
        ~MMapHandler();

        // Opens (or creates) a file and maps it into memory.
        // min_size: The initial size to allocate on disk.
        void open_file(const std::string& filepath, size_t min_size);

        // Closes the file and syncs data to disk.
        void close_file();

        // Expands the file size (needed when we add more vectors).
        // WARNING: This invalidates old pointers!
        void resize(size_t new_size);

        // Returns the raw pointer to the start of the memory block.
        // We cast this to (Node*) in the IndexManager.
        void* get_data() const;

        // Returns current file size.
        size_t get_size() const;

    private:
        std::string file_path_;
        size_t file_size_;
        void* data_; // The magic pointer. Accessing this reads/writes to disk.

        // OS-Specific Handles
#ifdef _WIN32
        void* file_handle_;  // HANDLE
        void* map_handle_;   // HANDLE
#else
        int file_fd_;        // File Descriptor
#endif
    };

} // namespace nanodb