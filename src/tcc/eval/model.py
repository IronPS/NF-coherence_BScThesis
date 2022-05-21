
import os
import numpy as np
from natsort import natsorted
from ..util import directory as dir

class ModelPaths():
    def __init__(self, model_path, fields=None):
        self.name_ = model_path.split(os.sep)[-1]

        self.paths_ = natsorted(dir.list_dir(model_path, dir.ListDirPolicyEnum.DIRS_ONLY))
        self.fields_ = {}
        for p in self.paths_:
            fname = p.split(os.sep)[-1].split('-')[0]
            self.fields_[fname] = p

        if fields:
            fields = np.array([fields]).flatten()
            fields_to_remove = [f for f in self.fields_.keys() if f not in fields]

            for f in fields_to_remove:
                del self.fields_[f]


    def __getitem__(self, fname):
        return self.fields_[fname]

    def __len__(self):
        return len(self.fields_.keys())

    def __iter__(self):
        return self.fields_.__iter__()

    def keys(self):
        return list(self.fields_.keys())

    def __str__(self):
        return "\'" + self.name_ + "\' " + str(self.fields_)
    
    def __repr__(self):
        def print_dots_f():
            return "\t\t⋮\n"

        print_dots = False

        str = "\'{}\' (\n".format(self.name_)
        items = np.array([i for i in self.fields_.items()])

        if items.shape[0] > 9:
            items_ = list(items[:3])
            items_ += list(items[-3:])
            items = items_
            print_dots = True

        for fname, fpath in items:
            str += "\t\'{}\' : \'{}\',\n".format(fname, fpath)
            if print_dots and i == 2:
                str += print_dots_f()
        str += ")\n"

        return str

class ModelFieldPaths():
    def __init__(self, model_name, field_path):
        self.model_name = model_name
        self.name_ = field_path.split(os.sep)[-1]

        self.paths_ = natsorted(dir.list_dir(field_path, dir.ListDirPolicyEnum.FILES_ONLY))
        
        self.atoms_ = {}
        for i, p in enumerate(self.paths_):
            # fname = p.split('/')[-1].split(self.model_name)[-1][1:]
            # fname = fname.split('.')
            # fname = fname[0] if len(fname) <= 1 else '.'.join(fname[:-1])
            self.atoms_[i] = p

    def __getitem__(self, index):
        return self.atoms_[index]

    def __len__(self):
        return len(self.atoms_.keys())

    def __iter__(self):
        return iter([(k,v) for k, v in self.atoms_.items()])

    def keys(self):
        return list(self.atoms_.keys())

    def __str__(self):
        return "\'" + self.name_ + "\' " + str(self.atoms_)
    
    def __repr__(self):
        def print_dots_f():
            return "\t\t⋮\n"

        print_dots = False

        str = "\'{}\' (\n".format(self.name_)
        items = np.array([i for i in self.atoms_.items()])

        if items.shape[0] > 9:
            items_ = list(items[:3])
            items_ += list(items[-3:])
            items = items_
            print_dots = True

        for i, (index, fpath) in enumerate(items):
            str += "\t{} : \'{}\',\n".format(index, fpath)
            # str += "\t\'{}\',\n".format(fname)
            if print_dots and i == 2:
                str += print_dots_f()
        str += ")\n"

        return str

class ModelFieldLoader(ModelFieldPaths):
    def __init__(self, model_name, field_path, load_fn=lambda x: x):
        super().__init__(model_name, field_path)
        self.load_fn_ = load_fn

    def set_load_function(self, load_fn):
        self.load_fn_ = load_fn
    
    def __getitem__(self, fname):
        return self.load_fn_(self.atoms_[fname])

    def __iter__(self):
        return map(lambda pair: (pair[0], self.load_fn_(pair[1])), super().__iter__())

class ModelLoader():
    def __init__(self, model_path, loaders_dict={}):
        self.model_paths_ = ModelPaths(model_path, fields=list(loaders_dict.keys()))
        self.name_ = self.model_paths_.name_
        self.fields_ = {}
        for p in self.model_paths_:
            mf = ModelFieldLoader(self.name_, self.model_paths_[p])
            self.fields_[mf.name_] = mf

        for k, f in loaders_dict.items():
            try:
                self.fields_[k].set_load_function(f)
            except:
                pass

    def field_names(self):
        return list(self.fields_.keys())

    def __getitem__(self, idx):
        return self.fields_[idx]

    def __len__(self):
        return len(list(self.fields_.keys()))

    def __repr__(self):
        str = "ModelLoader: '{}'".format(self.name_) + " {\n"
        for f in self.field_names():
            str += "\t'{}': {} file(s)\n".format(f, len(self.fields_[f]))
        str += "}"
        return str