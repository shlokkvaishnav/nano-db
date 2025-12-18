#include "../../include/storage/mmap_handler.hpp"
#include <iostream>
#include <filesystem>

// ---------------------------------------------------------
// OS-Specific Includes
// ---------------------------------------------------------
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace nanodb {

    MMapHandler::MMapHandler() 
        : file_size_(0), data_(nullptr) 
    {
#ifdef _WIN32
        file_handle_ = INVALID_HANDLE_VALUE;
        map_handle_ = NULL;
#else
        file_fd_ = -1;
#endif
    }

    MMapHandler::~MMapHandler() {
        close_file();
    }

    void MMapHandler::open_file(const std::string& filepath, size_t min_size) {
        file_path_ = filepath;
        file_size_ = min_size;

        // Create directory if it doesn't exist
        std::filesystem::path p(filepath);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

#ifdef _WIN32
        // ---------------- WINDOWS IMPLEMENTATION ----------------
        
        // 1. Open File
        file_handle_ = CreateFileA(
            filepath.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL
        );

        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Failed to open file (Windows)");
        }

        // 2. Resize File (if new or smaller than requested)
        LARGE_INTEGER size;
        GetFileSizeEx(file_handle_, &size);
        if (size.QuadPart < (long long)min_size) {
            size.QuadPart = min_size;
            // Set file pointer to end and write EOF
            SetFilePointerEx(file_handle_, size, NULL, FILE_BEGIN);
            SetEndOfFile(file_handle_);
        } else {
            file_size_ = size.QuadPart;
        }

        // 3. Create Mapping Object
        map_handle_ = CreateFileMappingA(
            file_handle_, NULL, PAGE_READWRITE, 0, 0, NULL
        );
        
        if (map_handle_ == NULL) {
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to create file mapping");
        }

        // 4. Map View
        data_ = MapViewOfFile(map_handle_, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if (data_ == NULL) {
            CloseHandle(map_handle_);
            CloseHandle(file_handle_);
            throw std::runtime_error("Failed to map view of file");
        }

#else
        // ---------------- LINUX / POSIX IMPLEMENTATION ----------------

        // 1. Open File
        file_fd_ = open(filepath.c_str(), O_RDWR | O_CREAT, 0666);
        if (file_fd_ == -1) {
            throw std::runtime_error("Failed to open file (POSIX)");
        }

        // 2. Resize File
        struct stat st;
        fstat(file_fd_, &st);
        if ((size_t)st.st_size < min_size) {
            if (ftruncate(file_fd_, min_size) != 0) {
                close(file_fd_);
                throw std::runtime_error("Failed to resize file");
            }
        } else {
            file_size_ = st.st_size;
        }

        // 3. Map Memory
        data_ = mmap(nullptr, file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, file_fd_, 0);
        if (data_ == MAP_FAILED) {
            close(file_fd_);
            throw std::runtime_error("mmap failed");
        }
#endif
    }

    void MMapHandler::close_file() {
        if (data_) {
#ifdef _WIN32
            UnmapViewOfFile(data_);
            CloseHandle(map_handle_);
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
#else
            munmap(data_, file_size_);
            close(file_fd_);
            file_fd_ = -1;
#endif
            data_ = nullptr;
        }
    }

    void MMapHandler::resize(size_t new_size) {
        // To resize an mmap, we must unmap, expand the file, and remap.
        // This is expensive, so we usually double the size when needed.
        close_file();
        open_file(file_path_, new_size);
    }

    void* MMapHandler::get_data() const {
        return data_;
    }

    size_t MMapHandler::get_size() const {
        return file_size_;
    }

} // namespace nanodb