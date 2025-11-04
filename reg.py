from __future__ import annotations
import pickle
import os
import wmi
from typing import List
import pprint as pp
from datetime import datetime
import json
import logging
class Registry:
    def __init__(self,**kwargs) -> None:
        self.logger = logging.getLogger("elsem")
        self.d_mac = {"expired":"no"}
        self.d_mac.update(kwargs)  
        self.save()     
    def save(self) -> None:
        try:
            if 'machine_name' not in self.d_mac and 'hdd' not in self.d_mac:
                self.d_mac["machine_name"] =  os.environ['COMPUTERNAME']
                self.d_mac['hdd'] = self.hdd_serial()
            d_bin = Registry.dict_to_binary(self.d_mac)           
            reg_file = f"{self.d_mac['machine_name']}.bin"
            with open(f"{reg_file}", 'wb') as handle:
                pickle.dump(d_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)           
        except Exception as err:
            self.logger.error(f"Couldn't save the registration file elsem_bin.reg:{err}")
    def update(self):
        reg_file = f"{self.d_mac['machine_name']}.bin"
        d_bin = Registry.dict_to_binary(self.d_mac)
        with open(f"{reg_file}", 'wb') as handle:
            pickle.dump(d_bin, handle, protocol=pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def restore(reg_file) -> Registry:
        with open(reg_file, 'rb') as handle:
            reg = pickle.load(handle)
            reg = Registry.binary_to_dict(reg)
            return Registry(**reg)
    def hdd_serial(self) -> str:
        c = wmi.WMI()
        for item in c.Win32_PhysicalMedia():
            if "PHYSICALDRIVE" in str(item.Tag).upper():
                serialNo = item.SerialNumber
                return str(serialNo)
        print(self)
    def __str__(self):
        return f"{self.d_mac['machine_name']}:\
            {self.d_mac['hdd'].strip()}:\
            {self.d_mac['expiry_date']}"
    
    def xcode(self,x: str) -> dict:
        self.d_mac['xcode'] = x
        return self.d_mac
    def check_xcode(self) -> bool:
        if "xcode" in self.d_mac:
            return self.d_mac['xcode']=="346B"
        else:
            return False
    def check_exp_date(self):
        exp_date = datetime.strptime(self.d_mac['expiry_date'], '%Y-%m-%d').date()
        if exp_date > datetime.now().date():
            return True
        else:
            return False
    def check(self) -> bool:
        self.logger.info(f"xcode: {self.check_xcode()}")
        self.logger.info(f"expired: {self.d_mac['expired']}")
        self.logger.info(f"machine name {self.d_mac['machine_name']} {os.environ['COMPUTERNAME']}")
        self.logger.info(f"hdd {self.d_mac['hdd']} {self.hdd_serial()}")
        self.logger.info(f"expiry date {self.check_exp_date()}")
        
        if all((self.check_xcode(),self.d_mac['expired'] == "no",
               self.d_mac["machine_name"]==os.environ['COMPUTERNAME'],
               self.d_mac['hdd']==self.hdd_serial() and self.check_exp_date())):
           return True
        else:
            return False
    @staticmethod
    def dict_to_binary(the_dict):
        str = json.dumps(the_dict)
        binary = ' '.join(format(ord(letter), 'b') for letter in str)
        return binary
    @staticmethod
    def binary_to_dict(the_binary):
        try:
            jsn = ''.join(chr(int(x, 2)) for x in the_binary.split())
            d = json.loads(jsn)  
            return d
        except Exception as err:
            print(f"failed in converting d_mac into binary: {err}")
if __name__=="__main__":
    d = Registry.restore('IMKORL-BRCJPS3.bin')
    print(d)
    d.xcode("346B")
    d.d_mac["expiry_date"] = "2024-12-31"
    d.d_mac["expired"] = "no"
    reg = Registry(**d.d_mac)
    reg.update()
    print(reg.check_xcode())
    print(reg.check_exp_date())
    pp.pprint(reg.d_mac)