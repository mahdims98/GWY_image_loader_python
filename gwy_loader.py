import struct
from collections import OrderedDict
import numpy as np
from six import BytesIO, string_types
from six.moves import range


class GwyObject(OrderedDict):
        """GwyObject."""

        def __init__(self, name, data=None, typecodes=None):
                OrderedDict.__init__(self)
                self.name = name
                self.typecodes = {}
                if isinstance(data, dict):
                        self.update(data)
                if isinstance(typecodes, dict):
                        self.typecodes.update(typecodes)

        def __str__(self):
                return '<GwyObject "{name}">({keys})'.format(
                        name=self.name,
                        keys=', '.join("'{}'".format(k) for k in self.keys())
                )

        @classmethod
        def frombuffer(cls, buf, return_size=False):
                pos = buf.find(b'\0')
                name = buf[:pos].decode('latin-1')
                size = struct.unpack('<I', buf[pos + 1:pos + 5])[0]
                object_data = buf[pos + 5:pos + 5 + size]
                buf = object_data
                data = OrderedDict()
                typecodes = {}
                while len(buf) > 0:
                        (component_name, component_data, component_typecode,
                         component_size) = component_from_buffer(buf, return_size=True)
                        data[component_name] = component_data
                        typecodes[component_name] = component_typecode
                        buf = buf[component_size:]
                try:
                        type_class = _gwyddion_types[name]
                        obj = type_class(data=data, typecodes=typecodes)
                except KeyError:
                        obj = GwyObject(name, data, typecodes=typecodes)
                if return_size:
                        return obj, len(name) + 5 + size
                return obj

        def serialize(self):
                io = BytesIO()
                for k in self.keys():
                        try:
                                typecode = self.typecodes[k]
                        except KeyError:
                                typecode = None
                        io.write(serialize_component(k, self[k], typecode))
                buf = io.getvalue()
                return b''.join([
                        self.name.encode('latin-1'),
                        b'\0',
                        struct.pack('<I', len(buf)),
                        buf
                ])

        @classmethod
        def fromfile(cls, file):
                if isinstance(file, string_types):
                        with open(file, 'rb') as f:
                                return GwyObject._read_file(f)
                return GwyObject._read_file(file)

        def tofile(self, file):
                if isinstance(file, string_types):
                        with open(file, 'wb') as f:
                                self._write_file(f)
                else:
                        self._write_file(file)

        @classmethod
        def _read_file(cls, f):
                data = f.read()
                assert data[:4] == b'GWYP'
                return cls.frombuffer(data[4:])

        def _write_file(self, f):
                f.write(b'GWYP')
                f.write(self.serialize())


class GwyContainer(GwyObject):
        def __init__(self, data=None, typecodes=None):
                super(GwyContainer, self).__init__('GwyContainer', data, typecodes)

        @property
        def filename(self):
                return self.get('/filename', None)

        @filename.setter
        def filename(self, name):
                self['/filename'] = name


class GwyDataField(GwyObject):
        def __init__(self, data,
                     xreal=1.0, yreal=1.0, xoff=0, yoff=0,
                     si_unit_xy=None, si_unit_z=None,
                     typecodes=None):
                super(GwyDataField, self).__init__('GwyDataField', typecodes=typecodes)
                if isinstance(data, OrderedDict):
                        self.update(data)
                else:
                        assert isinstance(data, np.ndarray) and len(data.shape) == 2
                        self.xreal, self.yreal = xreal, yreal
                        self.xoff, self.yoff = xoff, yoff
                        self.si_unit_xy, self.si_unit_z = si_unit_xy, si_unit_z
                        self.data = data
                self.typecodes.update({
                        'xres': 'i', 'yres': 'i',
                        'xreal': 'd', 'yreal': 'd',
                        'xoff': 'd', 'yoff': 'd',
                })

        @property
        def data(self):
                xres, yres = self['xres'], self['yres']
                return self['data'].reshape((yres, xres))

        @data.setter
        def data(self, new_data):
                assert isinstance(new_data, np.ndarray) and new_data.ndim == 2
                yres, xres = new_data.shape
                self['xres'], self['yres'] = xres, yres
                self['data'] = new_data.flatten()

        @property
        def xreal(self):
                return self.get('xreal', None)

        @xreal.setter
        def xreal(self, width):
                if width is None:
                        if 'xreal' in self:
                                del self['xreal']
                else:
                        self['xreal'] = width

        @property
        def yreal(self):
                return self.get('yreal', None)

        @yreal.setter
        def yreal(self, height):
                if height is None:
                        if 'yreal' in self:
                                del self['yreal']
                else:
                        self['yreal'] = height

        @property
        def xoff(self):
                return self.get('xoff', 0)

        @xoff.setter
        def xoff(self, offset):
                self['xoff'] = offset

        @property
        def yoff(self):
                return self.get('yoff', 0)

        @yoff.setter
        def yoff(self, offset):
                self['yoff'] = offset

        @property
        def si_unit_xy(self):
                return self.get('si_unit_xy', None)

        @si_unit_xy.setter
        def si_unit_xy(self, unit):
                if unit is None:
                        if 'si_unit_xy' in self:
                                del self['si_unit_xy']
                elif isinstance(unit, string_types):
                        self['si_unit_xy'] = GwySIUnit(unitstr=unit)
                else:
                        self['si_unit_xy'] = unit

        @property
        def si_unit_z(self):
                return self.get('si_unit_z', None)

        @si_unit_z.setter
        def si_unit_z(self, unit):
                if unit is None:
                        if 'si_unit_z' in self:
                                del self['si_unit_z']
                elif isinstance(unit, string_types):
                        self['si_unit_z'] = GwySIUnit(unitstr=unit)
                else:
                        self['si_unit_z'] = unit


class GwySIUnit(GwyObject):
        def __init__(self, data=None, unitstr='', typecodes=None):
                super(GwySIUnit, self).__init__('GwySIUnit', typecodes=typecodes)
                if isinstance(data, OrderedDict):
                        self.update(data)
                else:
                        self.typecodes['unitstr'] = 's'
                        self.unitstr = unitstr

        @property
        def unitstr(self):
                return self['unitstr']

        @unitstr.setter
        def unitstr(self, s):
                self['unitstr'] = s


def component_from_buffer(buf, return_size=False):
        """Interpret a buffer as a serialized component."""
        pos = buf.find(b'\0')
        name = buf[:pos].decode('latin-1')
        typecode = buf[pos + 1:pos + 2].decode('latin-1')
        pos += 2
        data = None
        endpos = pos
        if typecode == 'o':
                data, size = GwyObject.frombuffer(buf[pos:], return_size=True)
                endpos += size
        elif typecode == 's':
                endpos = buf.find(b'\0', pos)
                data = buf[pos:endpos].decode('latin-1')
                endpos += 1
        elif typecode == 'b':
                data = ord(buf[pos:pos + 1]) != 0
                endpos += 1
        elif typecode == 'c':
                data = buf[pos]
                endpos += 1
        elif typecode == 'i':
                data = struct.unpack('<i', buf[endpos:endpos + 4])[0]
                endpos += 4
        elif typecode == 'q':
                data = struct.unpack('<q', buf[endpos:endpos + 8])[0]
                endpos += 8
        elif typecode == 'd':
                data = struct.unpack('<d', buf[endpos:endpos + 8])[0]
                endpos += 8
        elif typecode in 'CIQD':
                numitems = struct.unpack('<I', buf[pos:pos + 4])[0]
                endpos += 4
                typelookup = {
                        'C': np.dtype('<S'), 'I': np.dtype('<i4'),
                        'Q': np.dtype('<i8'), 'D': np.dtype('<f8')
                }
                dtype = typelookup[typecode]
                pos, endpos = endpos, endpos + dtype.itemsize * numitems
                data = np.fromstring(buf[pos:endpos], dtype=dtype)
        elif typecode == 'S':
                numitems = struct.unpack('<I', buf[pos:pos + 4])[0]
                endpos += 4
                data = []
                for _ in range(numitems):
                        pos = endpos
                        endpos = buf.find(b'\0', pos)
                        data.append(buf[pos:endpos].decode('latin-1'))
                        endpos += 1
        elif typecode == 'O':
                numitems = struct.unpack('<I', buf[pos:pos + 4])[0]
                endpos += 4
                data = []
                for _ in range(numitems):
                        pos = endpos
                        objdata, size = GwyObject.frombuffer(buf[pos:], return_size=True)
                        data.append(objdata)
                        endpos += size
        else:
                raise NotImplementedError
        if return_size:
                return name, data, typecode, endpos
        return name, data, typecode


def guess_typecode(value):
        """Guess Gwyddion typecode for `value`."""
        if np.isscalar(value) and hasattr(value, 'item'):
                value = value.item()
        if isinstance(value, GwyObject):
                return 'o'
        elif isinstance(value, string_types):
                if len(value) == 1:
                        return 'c'
                else:
                        return 's'
        elif type(value) is bool:
                return 'b'
        elif type(value) is int:
                if abs(value) < 2 ** 31:
                        return 'i'
                else:
                        return 'q'
        elif type(value) is float:
                return 'd'
        elif type(value) is np.ndarray:
                t = value.dtype.type
                if t == np.dtype('f8'):
                        return 'D'
                elif t == np.dtype('i8'):
                        return 'Q'
                elif t == np.dtype('i4'):
                        return 'I'
                elif t == np.dtype('S'):
                        return 'C'
                else:
                        raise NotImplementedError
        else:
                raise NotImplementedError('{}, type: {}'.format(value, type(value)))


def serialize_component(name, value, typecode=None):
        """Serialize `value` to a Gwyddion component."""
        if typecode is None:
                typecode = guess_typecode(value)
        if typecode == 'o':
                buf = value.serialize()
        elif typecode == 's':
                buf = b''.join([value.encode('latin-1'), b'\0'])
        elif typecode == 'c':
                buf = value.encode('latin-1')
        elif typecode == 'b':
                buf = chr(value).encode('latin-1')
        elif typecode in 'iqd':
                buf = struct.pack('<' + typecode, value)
        elif typecode in 'CIQD':
                typelookup = {
                        'C': np.dtype('<S'), 'I': np.dtype('<i4'),
                        'Q': np.dtype('<i8'), 'D': np.dtype('<f8')
                }
                data = value.astype(typelookup[typecode]).data
                buf = b''.join([
                        struct.pack('<I', len(value)),
                        memoryview(data).tobytes()
                ])
        elif typecode == 'S':
                data = [struct.pack('<I', len(value)), ]
                data += [s.encode('latin-1') + b'\0' for s in value]
                buf = b''.join(data)
        elif typecode == 'O':
                data = [struct.pack('<I', len(value)), ]
                data += [obj.serialize() for obj in value]
                buf = b''.join(data)
        else:
                raise NotImplementedError('name: {}, typecode: {}, type: {}'
                                          .format(name, typecode, type(value)))
        return b''.join([
                name.encode('latin-1'), b'\0',
                typecode.encode('latin-1'),
                buf
        ])


def find_datafields(obj):
        """Return pairs of (number, title) for all available data fields."""
        token = '/data/title'
        channels = [int(k[1:-len(token)]) for k, _ in obj.items()
                    if k.endswith(token)]
        titles = [obj['/{}/data/title'.format(ch)] for ch in channels]
        return zip(channels, titles)


def get_datafields(obj):
        """Return a dictionary of titles and their corresponding data fields."""
        return {
                v: obj['/{chnum}/data'.format(chnum=k)]
                for k, v in find_datafields(obj)
        }


def load_gwy(file_or_filename):
        """Load a Gwyddion file and return the data fields."""
        obj = GwyObject.fromfile(file_or_filename)
        return get_datafields(obj)


def get_channels(file_or_filename):
        """Return a list of channel titles available in the Gwyddion file."""
        obj = GwyObject.fromfile(file_or_filename)
        return [title for _, title in find_datafields(obj)]


def get_metadata(file_or_filename):
        """Return a dictionary of metadata for each channel in the Gwyddion file."""
        obj = GwyObject.fromfile(file_or_filename)
        channels = dict(find_datafields(obj))
        
        metadata = {}
        for ch_num, title in channels.items():
                meta_key = '/{}/meta'.format(ch_num)
                if meta_key in obj:
                        metadata[title] = dict(obj[meta_key])
                else:
                        metadata[title] = {}
        return metadata


# Type lookup table
_gwyddion_types = {
        'GwyContainer': GwyContainer,
        'GwyDataField': GwyDataField,
        'GwySIUnit': GwySIUnit,
}

# Example usage
if __name__ == "__main__":
        # Load a .gwy file
        filename = '2023-12-01_16-05-17_G1_DDC_G2_DDC_6m_400m_0027_CALIBRATED.gwy'

        try:
                # List all available channels
                channels = get_channels(filename)
                print(f"Available channels: {channels}\n" + "-" * 40)
                
                # Read and print metadata for each channel
                metadata = get_metadata(filename)
                for title, meta in metadata.items():
                        print(f"Metadata for '{title}':")
                        for k, v in list(meta.items())[:20]: # Show first 5 items
                                print(f"  {k}: {v}")
                        print("  ...\n")
                print("-" * 40)
        except FileNotFoundError:
                print(f"Metadata test skipped: {filename} not found.")

        data_fields = load_gwy(filename)

        # Print available channels and their data
        for title, field in data_fields.items():
                print(f"Channel: {title}")
                print(f"Data shape: {field.data.shape}")
                print(f"Physical size: {field.xreal} x {field.yreal}")
                print("-" * 40)


        height_field = data_fields["Height [Fwd]"]
        width = height_field.xreal
        height = height_field.yreal
        data = height_field.data


        # Plot the data using matplotlib.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(data, interpolation='none', origin='upper',
                extent=(0, height_field.xreal, 0, height_field.yreal))
        plt.show()



# import gwyfile
#
# # Load a Gwyddion file into memory
# obj = gwyfile.load('test.gwy')
# # Return a dictionary with the datafield titles as keys and the
# # datafield objects as values.
# channels = gwyfile.util.get_datafields(obj)
#
# channel = channels["Height [Fwd]"]
# # Datafield objects have a `data` property to access their
# # two-dimensional data as numpy arrays.
# data = channel.data
#
# # Plot the data using matplotlib.
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.imshow(data, interpolation='none', origin='upper',
#         extent=(0, channel.xreal, 0, channel.yreal))
# plt.show()