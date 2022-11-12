from struct import unpack
import numpy as np

# constants from the file spec
RECORD_SIZE=826
RECORD_HEADER_SIZE=24
RECORD_ARRAY_SIZE=401

# verbosity values
VERBOSITY_ALL = 2    # print warnings and errors  
VERBOSITY_ERRORS = 1 # print errors
VERBOSITY_NONE = 0   # print nothing


class SRSRecord:
    """Holds one 826 byte SRS Record."""

    _site_to_name = {
        1: "Palehua",
        2: "Holloman",
        3: "Learmonth",
        4: "San Vito",
        # add new site names here ..
        }

    def __init__(self):
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        self.minute = None
        self.seconds = None

        self.site_number = None
        self.site_name = None
        self.n_bands_per_record = None

        self.a_start_freq = None
        self.a_end_freq = None
        self.a_num_bytes = None
        self.a_analyser_reference_level = None
        self.a_analyser_attenuation = None

        self.b_start_freq = None
        self.b_end_freq = None
        self.b_num_bytes = None
        self.b_analyser_reference_level = None
        self.b_analyser_attenuation = None

        # dictionary that maps frequency in mega hertz to level
        self.a_values = {}

        # dictionary that maps frequency in mega hertz to level
        self.b_values = {}        
        return


    def _parse_srs_file_header(self, header_bytes, verbosity = VERBOSITY_ALL):

        fields = unpack(
            #     General header information
            '>' #          (data packed in big endian format)
            'B' # 1        Year (last 2 digits)                   Byte integer (unsigned)
            'B' # 2        Month number (1 to 12)                    "
            'B' # 3        Day (1 to 31)                             "
            'B' # 4        Hour (0 to 23 UT)                         "
            'B' # 5        Minute (0 to 59)                          "
            'B' # 6        Second at start of scan (0 to 59)         "
            'B' # 7        Site Number (0 to 255)                    "
            'B' # 8        Number of bands in the record (2)         "

            # Band 1 (A-band) header information
            'h' # 9,10     Start Frequency (MHz)                  Word integer (16 bits)
            'H' # 11,12    End Frequency (MHz)                       "
            'H' # 13,14    Number of bytes in data record (401)      "
            'B' # 15       Analyser reference level               Byte integer
            'B' # 16       Analyser attenuation (dB)                 "

            # Band 2 (B-band) header information
            # 17-24         As for band 1
            'H'  # 17,18    Start Frequency (MHz)                 Word integer (16 bits)
            'H'  # 19,20    End Frequency (MHz)                       "
            'H'  # 21,22    Number of bytes in data record (401)      "
            'B'  # 23       Analyser reference level              Byte integer
            'B', # 24       Analyser attenuation (dB)                 "
            header_bytes)

        self.year = fields[0]
        self.month = fields[1]
        self.day = fields[2]
        self.hour = fields[3]
        self.minute = fields[4]
        self.seconds = fields[5]

        # read the site number and work out the site name
        self.site_number = fields[6]
        if self.site_number not in list(SRSRecord._site_to_name.keys()):

            # got an unknown site number.. complain a bit..
            if verbosity >= VERBOSITY_ALL:
                print(("Unknown site number: %s" % self.site_number))
                print("A list of known site numbers follows:")            
                for site_number, site_name in list(SRSRecord._site_to_name.items()):
                    print(("\t%s: %s" % (site_number, site_name)))

            # then set the site name to unknown.
            self.site_name = "UnknownSite"
        else:
            # otherwise look up the site using our lookup table
            self.site_name = SRSRecord._site_to_name[self.site_number]

        # read the number of bands
        self.n_bands_per_record = fields[7] # should be 2
        if self.n_bands_per_record != 2 and verbosity >= VERBOSITY_ERRORS:
            print(("Warning.. record has %s bands, expecting 2!" % self.n_bands_per_record))

        # read the a record meta data
        self.a_start_freq = fields[8]
        self.a_end_freq = fields[9]
        self.a_num_bytes = fields[10]
        if self.a_num_bytes != 401 and verbosity >= VERBOSITY_ERRORS:
            print(("Warning.. record has %s bytes in the a array, expecting 401!" %
                  self.a_num_bytes))

        self.a_analyser_reference_level = fields[11]        
        self.a_analyser_attenuation = fields[12]

        # read the b record meta data
        self.b_start_freq = fields[13]
        self.b_end_freq = fields[14]
        self.b_num_bytes = fields[15]
        if self.b_num_bytes != 401 and verbosity >= VERBOSITY_ERRORS:
            print(("Warning.. record has %s bytes in the b array, expecting 401!" %
                  self.b_num_bytes))
        self.b_analyser_reference_level = fields[16]
        self.b_analyser_attenuation = fields[17]        
        return


    def _parse_srs_a_levels(self, a_bytes):
        # unpack the frequency/levels from the first array
        for i in range(401):
            # freq equation from the srs file format spec
            freq_a = 25 + 50 * i / 400.0
            level_a = unpack('>B', a_bytes[i])[0]
            self.a_values[freq_a] = level_a
        return

    def _parse_srs_b_levels(self, b_bytes):
        for i in range(401):
            # freq equation from the srs file format spec
            freq_b = 75 + 105 * i / 400.0
            level_b = unpack('>B', b_bytes[i])[0]
            self.b_values[freq_b] = level_b
        return


    def __str__(self):
        return ("%s/%s/%s, %s:%s:%s"
                )% (
                    self.day, self.month, self.year, 
                    self.hour, self.minute, self.seconds,)

    def _dump(self, values):        
        freqs = list(values.keys())
        freqs.sort()
        #for freq in freqs:
            #print "%5s %s" % (freq, values[freq])
        return values

    def dump_a(self):
        values=self._dump(self.a_values)
        return values

    def dump_b(self):
        values=self._dump(self.b_values)
        return values


def read_srs_file(fname):
    """Parses an srs file and returns a list of SRSRecords."""

    # keep the records we read in here
    srs_records = []

    f = open(fname, "rb")
    while True:

        # read raw record data
        record_data = f.read(RECORD_SIZE)

        # if the length of the record data is zero we've reached the end of the data
        if len(record_data) == 0:
            break

        # break up the record bytes into header, array a and array b bytes
        header_bytes = record_data[:RECORD_HEADER_SIZE]
        a_bytes = record_data[RECORD_HEADER_SIZE : RECORD_HEADER_SIZE + RECORD_ARRAY_SIZE]
        b_bytes = record_data[RECORD_HEADER_SIZE + RECORD_ARRAY_SIZE :
                              RECORD_HEADER_SIZE + 2 * RECORD_ARRAY_SIZE]
        
        # make a new srs record
        record = SRSRecord()
        record._parse_srs_file_header(header_bytes, verbosity = VERBOSITY_ERRORS)
        record._parse_srs_a_levels(a_bytes)
        record._parse_srs_b_levels(b_bytes)
        srs_records.append(record)

    return srs_records


def main(srs_file):
    # parse the file.. (this is where the magic happens ;)
    srs_records = read_srs_file(fname=srs_file)
    timestamps=[str(srs_records[i]) for i in range(len(srs_records))]
    # play with the data
    final_data_a=[]
    final_data_b=[]
    for i in range(len(srs_records)):
        r0 = srs_records[i]
        value_a=r0.dump_a()
        value_b=r0.dump_b()
        final_data_a.append(value_a)
        final_data_b.append(value_b)
    final_data=[final_data_a,final_data_b,timestamps]
    return final_data
    
	
