from halide import *
import numpy as np
import ctypes
import platform
import imageio
import os
import cv2

class BufferStruct(ctypes.Structure):
    _fields_ = [
        #A device-handle for e.g. GPU memory used to back this buffer.
        ("dev", ctypes.c_uint64),

        #A pointer to the start of the data in main memory.
        ("host", ctypes.POINTER(ctypes.c_uint8)),

        #The size of the buffer in each dimension.
        ("extent", ctypes.c_int32 * 4),

        #Gives the spacing in memory between adjacent elements in the
        #given dimension.  The correct memory address for a load from
        #this buffer at position x, y, z, w is:
        #host + (x * stride[0] + y * stride[1] + z * stride[2] + w * stride[3]) * elem_size
        #By manipulating the strides and extents you can lazily crop,
        #transpose, and even flip buffers without modifying the data.
        ("stride", ctypes.c_int32 * 4),

        #Buffers often represent evaluation of a Func over some
        #domain. The min field encodes the top left corner of the
        #domain.
        ("min", ctypes.c_int32 * 4),

        #How many bytes does each buffer element take. This may be
        #replaced with a more general type code in the future. */
        ("elem_size", ctypes.c_int32),

        #This should be true if there is an existing device allocation
        #mirroring this buffer, and the data has been modified on the
        #host side.
        ("host_dirty", ctypes.c_bool),

        #This should be true if there is an existing device allocation
        #mirroring this buffer, and the data has been modified on the
        #device side.
        ("dev_dirty", ctypes.c_bool),
    ]


def buffer_t_to_buffer_struct(buffer):
    assert type(buffer) == Buffer
    b = buffer.raw_buffer()
    bb = BufferStruct()

    uint8_p_t = ctypes.POINTER(ctypes.c_ubyte)
    # host_p0 is the complicated way...
    #host_p0 = image_to_ndarray(Image(UInt(8), b)).ctypes.data
    # host_ptr_as_int is the easy way
    host_p = buffer.host_ptr_as_int()
    bb.host = ctypes.cast(host_p, uint8_p_t)
    #print("host_p", host_p0, host_p, bb.host)
    bb.dev = b.dev
    bb.elem_size = b.elem_size
    bb.host_dirty = b.host_dirty
    bb.dev_dirty = b.dev_dirty
    for i in range(4):
        bb.extent[i] = b.extent[i]
        bb.stride[i] = b.stride[i]
        bb.min[i] = b.min[i]
    return bb

def angle2rgb(angleValue):
    # It converts angle to normalized rgb color
    depth = 3
    rows,cols=angleValue.shape
    rgbValue = np.zeros((rows,cols,depth),dtype=np.float32)
    for i in range(0,rows):
        for j in range(0,cols):
            aV = angleValue[i,j]
            if aV >= 0 and aV < np.pi/2:
                rgbValue[i,j,0] = 1
                rgbValue[i,j,1] = np.sin(aV)
                rgbValue[i,j,2] = 0
            elif aV >= np.pi/2 and aV < np.pi:
                rgbValue[i,j,0] = np.cos(aV-np.pi/2)
                rgbValue[i,j,1] = 1
                rgbValue[i,j,2] = 0
            elif aV >= np.pi and aV < 3*np.pi/2:
                rgbValue[i,j,0] = 0
                rgbValue[i,j,1] = np.cos(aV-np.pi)
                rgbValue[i,j,2] = np.sin(aV-np.pi)
            elif aV >= 3*np.pi/2 and aV <= 2*np.pi:
                rgbValue[i,j,0] = np.sin(aV-3*np.pi/2)
                rgbValue[i,j,1] = 0
                rgbValue[i,j,2] = np.cos(aV-3*np.pi/2)

    return rgbValue

def main():
    # Read the whole video
    # filename = 'plaid.avi'
    # filename = 'grating.avi'
    # filename = '/home/lpzal1/mcgm_halide/Video_stimulus/bbaf2n.mpg'
    # filename = '/home/lpzal1/Databases/jokers/NINJA2_S001_S001_T002.MOV'
    filename = '/home/lpzal1/mcgm_halide/Video_stimulus/H2N2A.MP4'
    inVid = imageio.get_reader(filename,  'ffmpeg')
    fps = inVid.get_meta_data()['fps']
    im = inVid.get_data(0);
    noRow = im.shape[0]
    noCol = im.shape[1]

    noFrm = inVid.get_length()

    inVidData = np.ndarray(shape=(im.shape[0],im.shape[1],im.shape[2],inVid.get_length()),dtype=np.float32,order='F')
    for i,im in enumerate(inVid):
        inVidData[:,:,:,i] = np.float32(im)/255.0

    nargout = 2
    outVidData = list()

    for i in range(0,nargout):
        outVidData.append(np.zeros(shape=(im.shape[0],im.shape[1],inVid.get_length()),dtype=np.float32,order='F'))

    os.system("gcc -shared -o mcgmOpticalFlow_v02_halide.so mcgmOpticalFlow_v02_halide.o")
    mcgmOpticalFlow = ctypes.cdll.LoadLibrary("./mcgmOpticalFlow_v02_halide.so")
    assert mcgmOpticalFlow != None

    input_buf = Image(inVidData).buffer()
    input_buf_struct = buffer_t_to_buffer_struct(input_buf)
    input_buf_struct_p = ctypes.pointer(input_buf_struct)

    output_buf = list(); output_buf_struct = list(); output_buf_struct_p = list();
    for i in range(0,nargout):
        output_buf.append(Image(outVidData[i]).buffer())
        output_buf_struct.append(buffer_t_to_buffer_struct(output_buf[i]))
        output_buf_struct_p.append(ctypes.pointer(output_buf_struct[i]))

    noFrm = ctypes.c_uint8(inVid.get_length())
    # error = mcgmOpticalFlow.mcgmOpticalFlow_halide(input_buf_struct_p,noFrm,\
    #                                                output_buf_struct_p[0],\
    #                                                output_buf_struct_p[1],\
    #                                                output_buf_struct_p[2],\
    #                                                output_buf_struct_p[3],\
    #                                                output_buf_struct_p[4])

    filterthreshold = ctypes.c_float(1e-5)
    divisionthreshold = ctypes.c_float(1e-30)
    divisionthreshold2 = ctypes.c_float(0.99)
    speedthreshold = ctypes.c_float(1e-6)

    evalExpr = 'mcgmOpticalFlow.mcgmOpticalFlow_v02_halide(input_buf_struct_p,noFrm,filterthreshold,divisionthreshold,divisionthreshold2,speedthreshold'
    for i in range(0,nargout):
        evalExpr+=',output_buf_struct_p['+str(i)+']'
    evalExpr+=')'
    error=eval(evalExpr)

    if error:
        print("Failed !")
        return -1
    else:
        print("Success !")
        print("Write outputVidData to a video")
        # vid_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # vid_file = cv2.VideoWriter('d_spec.avi',vid_fourcc,25,(288,360),True)
        # for iF in range(0,outVidData.shape[3]):
        #     vid_file.write(outVidData[:,:,:,iF])
        # vid_file.release()

        # noChn = 3
        # rgbFrm = np.ones((noRow,noCol,noChn),dtype=np.uint8)
        outVid = imageio.get_writer('mcgmOpticalFlow_v02_h2n2a.avi',fps=fps)

        for iF in range(0,np.uint16(noFrm)):
            # Orientation Normalization
            speed = np.stack((outVidData[0][...,iF],outVidData[0][...,iF],outVidData[0][...,iF]),axis=2)
            # print('Median Speed: ',np.median(speed.ravel()))
            # print('Max Speed: ',speed.max())
            # print('Min Speed: ',speed.min())
            # cv2.imshow('speed',outVidData[0][...,iF])
            orientation = outVidData[1][...,iF] # Orientation
            orientation = orientation + (orientation>=2*np.pi)*(-2*np.pi)
            orientation = orientation + (orientation<=0)*(2*np.pi)

            normRgbFrame = angle2rgb(orientation)
            normRgbFrame = np.multiply(normRgbFrame,np.float32(np.absolute(speed) > speedthreshold))
            # normRgbFrame = np.multiply(normRgbFrame,np.float32(np.absolute(speed) < 23 ))
            # print('Median Orientation: ',np.median(orientation.ravel()))
            # print('Max Orientation: ',orientation.max())
            # print('Min Orientation: ',orientation.min())
            # cv2.imshow('orientation',cv2.cvtColor(normRgbFrame,cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            normRgbFrame = np.multiply(normRgbFrame,speed)

            rgbFrame = np.divide((normRgbFrame - normRgbFrame.min()),(normRgbFrame.max()-normRgbFrame.min())+10**-30)
            rgbFrame = np.uint8(np.round(np.multiply(rgbFrame,255.0)))
            outVid.append_data(rgbFrame)

        outVid.close()
        # outVid = imageio.get_writer('d_spec.avi',fps=fps)
        # for i in range(0,nargout):
        #     outVid = imageio.get_writer('output_'+str(i)+'.avi',fps=fps,quality=10)
        #     normOutVid = np.uint8((outVidData[i] - outVidData[i].min())/(outVidData[i].max()-outVidData[i].min())*255)
        #     for iF in range(0,outVidData[i].shape[3]):
        #         outVid.append_data(normOutVid[:,:,:,iF])
        #     outVid.close()
        return 0

if __name__ == "__main__":
    main()
