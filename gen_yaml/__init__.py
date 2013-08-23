#!/bin/env python

__authors__ = "Xavier Bouthillier" 
__contact__ = "xavier.bouthillier@umontreal.ca"

import traceback
import sys
import os
import re
import numpy as np
from collections import OrderedDict

__all__ = ["generate_params","write_files"]

def error():
    print """Try `python gen_yaml.py --help` for more information"""
    sys.exit(2)

generation_modes = {
    "log-uniform": 
        lambda hpmin, hpmax, hpnb :
            list(np.exp(np.arange(np.log(hpmin),
                                  np.log(hpmax)+(np.log(hpmax)-np.log(hpmin))/(hpnb-1.0),
                                  (np.log(hpmax)-np.log(hpmin))/(hpnb-1.0))))[:hpnb],
    "log-random-uniform": 
        lambda hpmin, hpmax, hpnb : 
            list(np.exp(np.random.uniform(np.log(hpmin),
                                          np.log(hpmax),hpnb))),
    "uniform":
        lambda hpmin, hpmax, hpnb :
            list(np.arange(hpmin,
                           hpmax+(hpmax-hpmin)/(hpnb-1.0),
                           (hpmax-hpmin)/(hpnb-1.0)))[:hpnb],
    "random-uniform":
        lambda hpmin, hpmax, hpnb : 
            list(np.random.uniform(hpmin,hpmax,hpnb)),
}

class HparamReader():
    def __init__(self,file_name):
        self.i = iter(filter(lambda a: a.strip(" ")[0]!="#" and a.strip(" ").strip("\n")!="",
                             open(file_name,'r').readlines()))

    def __iter__(self):
        return self

    def next(self):
        return self.build_hparam(self.i.next())

    def build_hparam(self,line):
        s_line = filter(lambda a:a.strip(' ')!='',line.split(' '))
        s_line = sum([k.split(',') for k in s_line],[])
        s_line = filter(lambda a:a.strip(' ')!='',[s.strip("\n").strip("\t").strip(",") for s in s_line])

#        if len(s_line)!=6:
#            print "Incorrect hyper-parameter configuration"
#            print line.strip("\n")
#            print "# Hyper-parameters :: min :: max :: how much :: generation-mode :: default value"
#            error()
        s_line = list(reversed(s_line))

        d = OrderedDict(hparam=s_line.pop())

        # look for list
        if s_line[-1][0]=="[":
            name, values = self._get_function(s_line,['[',']'])
        # get a generator
        else:
            name, values = self._get_function(s_line,['(',')'])
            print values
            values = list(make_hparams(*(values+[name])))

#        print values
        d['values'] = values
        d['default'] = self._convertobj(s_line.pop())

        # combinations
        try:
            d['comb_type'] = self._get_function(s_line)
        except ValueError:
            d['comb_type'] = ('alone',)
            
        return d

    def _get_function(self,s_line,delims=['(',')']):
        index = s_line[-1].index(delims[0])
        name = s_line[-1][:index]
        s_line[-1] = s_line[-1][index:]

        params = []
        cont = True
        while cont:
            if s_line[-1][-1]==delims[1]:
                cont = False
            if s_line[-1] not in delims:
                param = s_line.pop()
                for delim in delims:
                    param = param.strip(delim)
                params.append(self._convertobj(param))
            else:
                s_line.pop()

        return (name,params)

    def _convertobj(self,obj):
        if obj in ['True','False']:
            return obj=='True'

        try:
            return int(obj)
        except:
            obj = obj

        try:
            return float(obj)
        except:
            obj = obj

        return obj

def randomsearch(hparamfile,generate):
    """
        Generate
    """

    hparams = OrderedDict()
    hpnbs = []

    for hparam in HparamReader(hparamfile):
        
        hpnbs.append(hparam['hpnb'])

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        hparam.pop('default')

#        hparam['hpnb'] = max(hpnb,hparam['hpnb'])

        if "random" not in hparam["generate"]:
            print "*** Warning ***"
            print "    Hyperparameter",hparam["hparam"],": Random search, Generation function =", generate
            print "    Random search but not a random value generation? Are you sure that's what you want?"

        name = hparam.pop("hparam")
        hparams[name] = hparams.get(name,[]) + list(make_hparams(**hparam))

    rand = []
    while len(rand) < min(hpnbs):
        r = int(np.random.rand(1)*min(hpnbs))
        if r not in rand:
            rand.append(r)

    rand = np.array(rand)/(.0+min(hpnbs))

    values = [np.array(hparam)[list(rand*len(hparam))] for hparam in hparams.values()]

    return hparams.keys(), np.transpose(np.array(values))

def fixgridsearch(hparamfile,generate):

    hparams = OrderedDict()
    dhparams = OrderedDict()

    for hparam in HparamReader(hparamfile):

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        dhparams[hparam['hparam']] = hparam.pop("default")

        name = hparam.pop("hparam")
        hparams[name] = hparams.get(name,[]) + list(make_hparams(**hparam))

    values = np.zeros((sum([len(hparam) for hparam in hparams.values()]),len(hparams.keys())))

    j = 0
    for i, hparam in enumerate(hparams.items()):
        # set all default values
        values[j:j+len(hparam[1])] = np.array(dhparams.values())
        # set the value of the current hyper-parameter
        values[j:j+len(hparam[1]),i] = np.array(hparam[1])

        j += len(hparam[1])

    return hparams.keys(), values

# http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
    1-D arrays to form the cartesian product of.
    out : ndarray
    Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
    2-D array of shape (M, len(arrays)) containing cartesian products
    formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def fullgridsearch(hparamfile,generate):

    hparams = OrderedDict()
    dhparams = OrderedDict()

    for hparam in HparamReader(hparamfile):

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        hparam.pop("default")

        name = hparam.pop("hparam")

        hparams[name] = hparams.get(name,[]) + list(make_hparams(**hparam))

    return hparams.keys(), cartesian(hparams.values())
 

search_modes = {"random-search":randomsearch,
		"fix-grid-search":fixgridsearch,
		"full-grid-search":fullgridsearch}

def write_files(template,hpnames,hpvalues,save_path,force=False):
#    template = "".join(open(template,'r'))
    save_path = re.sub('.yaml$','',save_path)
    print hpnames

    files = []

    if len(hpvalues)>40:
        a = ""
        while a not in ["y","n"]:
            a = raw_input("Do you realy want to produce as much as %d yaml files? (y/n) " % len(hpvalues))
            if a=='n':
                sys.exit(0)


    # save templates
    for i, hparams in enumerate(hpvalues):
        file_name = '%(save_path)s%(i)d' % {"save_path":save_path,"i":i}
        
        d = dict(zip(hpnames,hparams))

        d.update({'save_path':file_name})

        try:
            tmp_template = template % d
        except KeyError as e:
            print "The key %(e)s is not present in hyper-parameter file" % {'e':e}
            error()

        file_name += '.yaml'

        if os.path.exists(file_name) and not force:
            print """file \"%(file)s\" already exists. 
Use --force option if you wish to overwrite them""" % {"file":file_name}
            error()
        else:
            f = open(file_name,'w')
            f.write(tmp_template)
            f.close()
        
        d.pop('save_path')
        files.append("%(file_name)s == %(hparams)s" % {"file_name":file_name,
                     "hparams":', '.join([str(v) for v in d.items()])})

    f = open(save_path+".index",'w')
    f.write('\n'.join(files)+'\n')
    f.close()

    return [f.split(" == ")[0] for f in files]

def get_closest_interval_type(values):
    """
        works with in-order and reversed values
    """
    if len(values)<2:
        return 'uniform'

    print values[0],values[-1],len(values)
    log_int = generation_modes['log-uniform'](values[0],values[-1],len(values))[:len(values)]
    normal = generation_modes['uniform'](values[0],values[-1],len(values))[:len(values)]

    if len(log_int)==0:
        log_int = np.zeros(len(values))
    if len(normal)==0:
        normal = np.zeros(len(values))
    
    print 
    print values
    print log_int
    print normal

    log_dist = np.array(values)-np.array(log_int)
    normal_dist = np.array(values)-np.array(normal)

    print log_dist
    print normal_dist

    if np.sum(log_dist*log_dist) < np.sum(normal_dist*normal_dist):
        return 'log'
    else:
        return 'normal'

def generate_params(hparamfile,generate,search_mode):
    # generate and search_mode are useless now 
    
    hparams = OrderedDict()
    dhparams = OrderedDict()
    names = []

    for hparam in HparamReader(hparamfile):

#        dhparams[hparam['hparam']] = hparam.pop("default")

#        hparams[name] = hparams.get(name,[]) + hparam['values']
#        hparams.append(hparam)
        hparam['indices'] = []
        hparams[hparam['hparam']] = hparam

#    values = np.zeros((sum([len(hparam) for hparam in hparams.values()]),len(hparams.keys())))
#    values = [[] for name in hparams.keys()]
    values = []
    min_v = []
    max_v = []
    intervals = []
    for hparam in hparams.values():
        print hparam['hparam'],
        try:
            float(hparam['values'][0])
            min_v.append(min(hparam['values']))
            max_v.append(max(hparam['values']))
            intervals.append(get_closest_interval_type(hparam['values']))
        except ValueError as e:
            print e
            min_v.append(0)
            max_v.append(max(hparam['values']))
            intervals.append(None)
        print intervals[-1]

    nbs = [len(hparam['values']) for hparam in hparams.values()]

    # first build alone params
    for name, hparam in filter(lambda a: a[1]['comb_type'][0]=='alone',hparams.items()):
        start = len(values)
        for value in hparam['values']:
#            values[hparams.keys().index(name)].append(value)
            v = [o_hparam['default'] for o_name, o_hparam in hparams.items()]
            v[hparams.keys().index(name)] = value
            if not contains(v,values,min_v,max_v,nbs,intervals):
                values.append(v)
#            for o_name, o_hparam  in filter(lambda a:a[0]!=name, hparams.items()):
#                values[hparams.keys().index(o_name)].append(o_hparam['default'])

        stop = len(values)
        hparam['indices'].append((start,stop))

    # build zip params
    for name, hparam in filter(lambda a: a[1]['comb_type'][0]=='zip',hparams.items()):
        start = len(values)
        comb_names = hparam['comb_type'][1]
#        print comb_names

        if len(comb_names) < 2:
            continue

        for comb_values in zip(*[hparams[n]['values'] for n in comb_names]):

            v = [o_hparam['default'] for o_name, o_hparam in hparams.items()]
            for comb_name, value in zip(comb_names,comb_values):
#                print comb_name, value
                v[hparams.keys().index(comb_name)] = value
            if not contains(v,values,min_v,max_v,nbs,intervals):
                values.append(v)
#            for o_name, o_hparam in filter(lambda a:a[0] not in comb_names,hparams.items()):
#                values[hparams.keys().index(o_name)].append(o_hparam['default'])

        stop = len(values)

        # remove zip in those zipped
        for comb_name in comb_names:
            hparams[comb_name]['indices'].append((start,stop))
            o_comb = hparams[comb_name]['comb_type'][1]
            for o_comb_name in o_comb:
                del o_comb[o_comb.index(o_comb_name)]


    # build cartesian products
        # first cartesian products on specific hparams
    combines = filter(lambda a: a[1]['comb_type'][0]=='combine',hparams.items())
    for name, hparam in filter(lambda a: a[1]['comb_type'][1][0]!='all',combines):
        #
#        print name
        comb_names = hparam['comb_type'][1]
#        print comb_names
        for comb_name in comb_names:
            comb_hparam = hparams[comb_name]
            if len(comb_hparam)==0:
                raise ValueError("""Cannot combine to a specific hyper-parameter that is combined also.
                The specific hyper-parameter needs to be zip() or alone""")

#            print comb_hparam['indices']
            for start, stop in comb_hparam['indices']:
                hp_values = values[start:stop]
#                print len(hp_values)

                for row in hp_values:
#                    print row
                    for value in hparam['values']:
#                        print value, len(values)
                        v = row[:]
                        v[hparams.keys().index(name)] = value
#                        print len(values)
                        if not contains(v,values,min_v,max_v,nbs,intervals):
                            values.append(v)
#                        for i, o_value in enumerate(row):
#                            if i!=hparams.keys().index(name):
#                                values[i].append(o_value)
#                            else:
#                                values[i].append(value)

#        if len(hparam['indices']):
#            (start, stop) = hparam['indices'][0]
#            print name, hparam['indices']
#            print zip(*values)[start:stop]


    for name, hparam in filter(lambda a: a[1]['comb_type'][1][0]=='all',combines):
        value_index = hparams.keys().index(name)
        for row in values:
            for value in hparam['values']:
                comb_values = list(row[:])
                comb_values[value_index] = value
                if not contains(comb_values,values,min_v,max_v,nbs,intervals):
                    values.append(comb_values)
#        pass
        
#    j = 0
#    for i, hparam in enumerate(hparams.items()):
#
#        print hparam
#        # set all default values
#        values[j:j+len(hparam[1])] = np.array(dhparams.values())
#        # set the value of the current hyper-parameter
#        values[j:j+len(hparam[1]),i] = np.array(hparam[1])
#
#        j += len(hparam[1])

#    for name, hparam in hparams.items():
#        if len(hparam['indices']):
#            (start, stop) = hparam['indices'][0]
#            print name, hparam['indices']
#            print zip(*values)[start:stop]

    print "show values"
    for row in values:
        print row

    return hparams.keys(), values

def contains(a,values,min,max,nbs,intervals):
#    a = np.array(a)
#    values = np.array(values)
#    min = np.array(min)
#    max = np.array(max)
#    nbs = np.array(nbs)
#    print min,max
    for value in values:
#        if a[2]==100.00000000000004 and a[0] and a[3]==0.05: 
#        print a[0], a[0]-1e-8
#        if abs(a[0]-1e-8) < 1e-15 and abs(value[0]-1e-8) < 1e-15 and a[2]==10 and value[2]==10:
#            print "hoooooooy\n\n"
#            print a, value
#            for i in range(len(a)):
#                print a[i]
#                print value[i]
#                print np.abs(a[i]-value[i])
#                print max[i]
#                print min[i]
#                print nbs[i]
#                print (max[i]-min[i])/(nbs[i]*100.0)
#                print np.abs(a[i]-value[i]) > (max[i]-min[i])/(nbs[i]*100.0)
#                print equal(a,value,min,max,nbs,intervals)
        if equal(a,value,min,max,nbs,intervals):
            return True

    return False

def equal(a,b,min,max,nbs,intervals):
    epsylon = [(i[1]-i[0])/(i[2]*100.0) for i in zip(min,max,nbs)]
    for i, (ai, bi) in enumerate(zip(a,b)):
#        print i, ai, bi
        try: # float, int, boolean
            float(ai),float(bi)
            if intervals[i]=='log':
                epsylon = (np.log(max[i])-np.log(min[i]))/(nbs[i]*100.0)
                diff = np.log(ai)-np.log(bi)
            elif intervals[i]=='normal':
                epsylon = (max[i]-min[i])/(nbs[i]*100.0)
                diff = ai-bi
            else:
                raise ValueError("Invalid interval type \'%s\'" % intervals[i])
#            print ai,bi,epsylon
            if abs(diff) > epsylon:
                return False
        except: # string
            if ai!=bi:
                return False

    return True
#    return np.all(np.abs(a-b) < epsylon)
#    return hparams.keys(), values


#        if "generate" not in hparam or hparam["generate"] in ["default",""]:
#            if hparam["generate"]=="":
#                print "*** Warning ***"
#                print "    Hyperparameter",hparam["hparam"]
#                print "    Please set generation mode : default"
#


#    try:
#        return search_modes[search_mode](hparamfile,generate)
#    except KeyError as e:
#        print "invalid search function : ",search_mode
#        print "Try ",", ".join(search_modes.keys())
#        error()

def make_hparams(hpmin,hpmax,hpnb,generate):
    try:
        return generation_modes[generate](hpmin,hpmax,hpnb)
    except KeyError as e:
        print "invalid generative function : ",generate
        print "Try ",", ".join(generation_modes.keys())
        error()
