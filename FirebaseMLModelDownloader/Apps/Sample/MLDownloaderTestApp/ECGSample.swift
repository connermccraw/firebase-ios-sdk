//
//  ECGSample.swift
//  MLDownloaderTestApp
//
//  Created by Conner McCraw on 4/24/22.
//

import Foundation

struct ECGSample: Decodable {
    var voltages: [Double]
    
    enum CodingKeys: String, CodingKey {
        case voltages
    }
}
