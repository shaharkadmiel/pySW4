"""
- dictionaries_and_dtypes.py-

Some dictionaries and data_types for WPP and SW4 image files

By: Omri Volk & Shahar Shani-Kadmiel, June 2015, kadmiel@post.bgu.ac.il

"""

import numpy as np

SW4_header_dtype = np.dtype([
        ('precision'         , 'int32'   ), #  4
        ('number of patches' , 'int32'   ), #  4
        ('time'              , 'float64' ), #  8
        ('plane'             , 'int32'   ), #  4
        ('coordinate'        , 'float64' ), #  8
        ('mode'              , 'int32'   ), #  4
        ('gridinfo'          , 'int32'   ), #  4
        ('creation time'     , 'S25'     )  # 25
                                            #----
                                            # 61
    ])

SW4_patch_dtype = np.dtype([
        ('h'         , 'float64' ), #  8
        ('zmin'      , 'float64' ), #  8
        ('ib'        , 'int32'   ), #  4
        ('ni'        , 'int32'   ), #  4
        ('jb'        , 'int32'   ), #  4
        ('nj'        , 'int32'   ), #  4
                                    #----
                                    # 32
    ])

#             mode # :  (mode                                     , unit      ),
SW4_mode_dict = { 1  :  ('ux'                                     , 'm'       ),
                  2  :  ('uy'                                     , 'm'       ),
                  3  :  ('uz'                                     , 'm'       ),
                  4  :  ('rho'                                    , 'km/m^3'  ),
                  5  :  ('lambda'                                 , 'Pa'      ),
                  6  :  ('mu'                                     , 'Pa'      ),
                  7  :  ('Vp'                                     , 'm/s'     ),
                  8  :  ('Vs'                                     , 'm/s'     ),
                  9  :  ('ux ex'                                  , 'm'       ),
                 10  :  ('uy ex'                                  , 'm'       ),
                 11  :  ('uz ex'                                  , 'm'       ),
                 12  :  ('div(u)'                                 , None      ),
                 13  :  ('curl(u)'                                , None      ),
                 14  :  ('div(du/dt)'                             , 's^-1'    ),
                 15  :  ('curl(du/dt)'                            , 's^-1'    ),
                 16  :  ('lat'                                    , 'degrees' ),
                 17  :  ('lon'                                    , 'degrees' ),
                 18  :  ('topo'                                   , 'm'       ),
                 19  :  ('x'                                      , 'm'       ),
                 20  :  ('y'                                      , 'm'       ),
                 21  :  ('z'                                      , 'm'       ),
                 22  :  ('ux error'                               , None      ),
                 23  :  ('uy error'                               , None      ),
                 24  :  ('uz error'                               , None      ),
                 25  :  ('|du/dt|'                                , 'm/s'     ),
                 26  :  ('sqrt((dux/dt)^2 + (duy/dt)^2)'          , 'm/s'     ),
                 27  :  ('max_t (sqrt((dux/dt)^2 + (duy/dt)^2))'  , 'm/s'     ),
                 28  :  ('max_t |duz/dt|'                         , 'm/s'     ),
                 29  :  ('|u|'                                    , 'm'       ),
                 30  :  ('sqrt(ux^2 + uy^2)'                      , 'm'       ),
                 31  :  ('max_t sqrt(ux^2 + uy^2)'                , 'm'       ),
                 32  :  ('max_t |uz|'                             , 'm'       )
                 }

SW4_plane_dict = {0  :  'x',
                  1  :  'y',
                  2  :  'z'}

prec_dict = {4   :   np.float32,
             8   :   np.float64
             }

