#ifndef MODULE_H
#define MODULE_H

#include <stdint.h>
#include <string>

namespace ABCS
{
   typedef enum
   {
      TYPE_TRANSMITTER,
      TYPE_RECEIVER,
      TYPE_BOTH
   } module_type;

   class Module // ABC
   {
      public:
      Module(std::string _name, module_type _type) :
         type(_type),
         module_name(_name)
      {}
      virtual ~Module() {}
      virtual int Init() = 0;
      std::string GetModuleName(void) { return module_name; }
      std::string GetInstanceName(void) { return instance_name; }
      std::string GetWholeName(void)
      {
         return module_name + ":" + instance_name;
      }

      static const char* const typeStr[3];

      protected:
      module_type type                  = TYPE_BOTH;
      uint32_t    current_host_memory   = 0;
      uint32_t    current_device_memory = 0;
      std::string module_name("");
      std::string instance_name("");
   };

}; // namespace ABCS

#endif /* MODULE_H */