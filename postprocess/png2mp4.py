def images_to_movie(input_folder,output_folder='same',output_name='movie',input_frames_per_second=1):

    import subprocess as sub
    import time

    print 'converting sequencial png files to mp4...\n'
    
    if input_folder[-1]!='/':input_folder+='/'
    if output_folder=='same':output_folder=input_folder
    if output_folder[-1]!='/':output_folder+='/'

    
    t6 = time.time()
    
    command = '''ffmpeg -y -pattern_type glob -r %f -i %s*.png '''%(input_frames_per_second,input_folder)
    command += '''-vcodec libx264 -r 25 -crf 25 -pass 1 -vb 6M -pix_fmt yuv420p '''
    command += '''-vf scale=trunc(iw/2)*2:trunc(ih/2)*2 -an %s%s.mp4'''%(output_folder,output_name)

    command = command.split()

    print '***\ncalling %s with the following arguments:\n' %command[0]
    for item in command[1:]:
        print item,
    print '\n***\n'

    time.sleep(1)

    p = sub.Popen(command)
    p.wait()

    print '\n******\nconvertion took %f seconds' %(time.time()-t6)

    
    
                     


