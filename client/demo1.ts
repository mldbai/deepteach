//
// demo1.ts
// Hao Deng, 2016-01-11
// Copyright (c) 2016 Datacratic Inc. All rights reserved.

// This code is written with trying new technology in mind ( virtual-dom),
// and not necessarily the simplest solution.
// In fact, using html directly would be easier.

/// <reference path='./typings/browser.d.ts' />

import $ = require('jquery')
import _ = require('lodash')
import V = require('virtual-dom')
let {h, diff, patch, create} = V
declare var require: (s: string) => any;
let foo = require('jquery-ui')
let _VText = require('virtual-dom/vnode/vtext')
import pie = require('./pie')

function VText(t: string){
    return addTags(new _VText(t))
}

// A pair of Id and distance
type IdDist=[string, number]
interface GroupResult{
    exploit : IdDist[],
    explore : IdDist[],
    prev    : IdDist[],
}

// The response from /similar
interface SimilarResponse {
    a: GroupResult,
    b: GroupResult,
    ignore: IdDist[],
    sample: IdDist[],
    deploy_id: string,
}

function SimilarResponse2State(s: SimilarResponse) : State{
    return {
        samples : s.sample,
        ignore  : s.ignore,
        a       : s.a.prev.concat(s.a.exploit),
        b       : s.b.prev.concat(s.b.exploit),
        maybeA  : s.a.explore,
        maybeB  : s.b.explore,
    } as State
}

function sendSimilar(deploy: boolean){
    $("#spinner").show()

    let f = (id: string ) => $(`#${id}  img`).map((idx:number, o: Element) => o.id).get()
    let data = {a: f('panelA'), b: f('panelB'), ignore: f('panelI')}

    let dataset = QueryString['dataset'];
    let prefix = QueryString['prefix'];
    let url= `../similar?dataset=${dataset}&prefix=${prefix}&deploy=${deploy}&input=${JSON.stringify(data)}`
    let w = deploy ? window.open('rt_prediction.html') : null
    $.ajax(url).done((ret: SimilarResponse) =>{
        let s = SimilarResponse2State(ret)
        let u = ui(s)
        let cs = document.body.children
        document.body.replaceChild(create(u), $('#main')[0])
        InitSortable()
        $("#spinner").hide()
        if(deploy){
            w.location.assign(`rt_prediction.html?deploy_id=${ret.deploy_id}`)
        }
    })
}

function onClick(evt: MouseEvent){
    sendSimilar(false)
}

function onDeploy(evt: MouseEvent){
    sendSimilar(true)
}

function addAllToA(evt: MouseEvent){
    $('#panelMaybeA > span').appendTo('#panelA')
}

function addAllToB(evt: MouseEvent){
    $('#panelMaybeB > span').appendTo('#panelB')
}

type Row = [string]

interface State{
    samples : IdDist[]
    ignore  : IdDist[]
    a       : IdDist[]
    maybeA  : IdDist[]
    b       : IdDist[]
    maybeB  : IdDist[]
}


const Img = ([imgId, dist]: IdDist) => {
    let dataset = QueryString['dataset'];
    let prefix = QueryString['prefix'];

    let img = h(`img#${imgId}`, {
        src: `${prefix}/${dataset}/${imgId}.jpg`,
        id: imgId,
        style: {
            margin: "4px", borderRadius: "25%", maxWidth: "200px",
        }
    } ,[] )
    let color = HSVtoRGB(dist * 0.333, 1, 1)
    let p = pie(dist, color)
    return h('span', {style:{}}, [img, p])

}

function Panel(id: string, style: any, idDists: IdDist[], title: string){
    let o1 = idDists.map(Img)
    let o = addTags(o1)
    return o.DIVp({id: id, className: 'sortable', style: style})
}

// Add HTML tag as a function into array, so for an array like a = ['abc', 'def']
// a.TD = TD(a)
// a.mapTD = [TD('abc', TD('def'))]
// a.TDp(prop: dict) = TD(a, prop)
// a.mapTDp(prop: dict) = [TD('abc', prop), TD('def', prop)]
// where TD is h('td', ...) from hyperscript
function addTags(o: any): any{
    if(o.hasOwnProperty('TD')){
        return o
    }
    const add = (tag: string) => {
        let r: any = {}
        r[tag] = { get: function() { return addTags(h(tag, this)) } }
        r[tag + 'p'] = {
            value: function(p: VirtualDOM.createProperties) {
                return addTags(h(tag, p, this))
            }
        }

        r['map' + tag] = {
            get: function() {
                return addTags(this.map(function(e: any) {
                    return h(tag, e)
                }))
            }
        }
        r['map' + tag + 'p'] = {
            value:  function(p: VirtualDOM.createProperties) {
                return addTags(this.map(function(e: any) {
                    return h(tag, p, e)
                }))
            }
        }
        return r
    }

    'TD TR TABLE H1 H2 H3 H4 H5 H6 DIV'.split(' ').forEach((tag: string) =>
        Object.defineProperties(o, add(tag)))
    return o
}


let btnStyle = {float: 'right', margin: '5px 30px'}
const createButton = (f: any) =>
    h('button',
        { "onclick":f
        , className: 'btn btn-info'
        , style: btnStyle
        }
      , "Add All")

// This function creates the whole ui
function ui (s: State){
    let pStyle = {
        minHeight: "300px",
        width: "48vw",
        border: "2px solid #ccc",
        borderRadius: "25px",
        borderCollapse: "separate",
        margin: "0 10px",
        display: "inline-block",
        height: "100%",
        padding: "10px"
    }

    let sampleStyle = _.clone(pStyle)
    sampleStyle['width'] = '96vw'

    //This modify array's prototype.
    addTags(Array.prototype)

    let aStyle = { maxHeight: "500px", overflow: "auto", marginBottom: "15px", minHeight: "100px"}
    let pa = Panel('panelA', pStyle, s.a, 'A')
    let pb = Panel('panelB', pStyle, s.b, 'B')
    let pa1 = Panel('panelMaybeA', pStyle, s.maybeA, 'Maybe A')
    let pb1 = Panel('panelMaybeB', pStyle, s.maybeB, 'Maybe B')

    let ps = Panel('panelSamples', sampleStyle, s.samples, 'Samples')
    //let pi = Panel('panelI', pStyle, s.ignore, 'Ingore')

    let btn       = addTags( h('button', {"onclick": onClick, className: 'btn btn-primary', style: btnStyle }, "Find Similar"))
    let btnDeploy = addTags( h('button', {"onclick": onDeploy, className: 'btn', style: btnStyle }, "Deploy"))
    let btn2 = createButton(addAllToA)
    let btn3 = createButton(addAllToB)
    let h2p = {style: { textAlign: "center", fontSize: "30px", color: "blue"}}
    let c2 = {colSpan: 2}
    let c3 = {colSpan: 3}
    let c4 = {colSpan: 4}

    return h('table#main',
        [ ([VText('Samples')] as any).mapDIVp(h2p).mapTDp(c4).TR
        , ([ps] as any).mapTDp(c4).TR
        , ([VText('A').DIVp(h2p), VText(""), VText('B').DIVp(h2p), [btn, btnDeploy]] as any).mapTD.TR,
        , ([pa, pb] as any).mapTDp(c2).TR
        , ([VText('Maybe A').DIVp(h2p), btn2, VText('Maybe B').DIVp(h2p), btn3] as any).mapTD.TR,
        , ([pa1, pb1] as any).mapTDp(c2).TR
        ])
}

function rows2State(rows: Row[]): State{
    return {
        samples: rows.map(row => [row[0], 0]),
        ignore: [],
        a: [],
        b: [],
        maybeA: [],
        maybeB: []
    } as State
}

function handleReceive(event: JQueryEventObject, ui: JQueryUI.SortableUIParams){
    let sid = ui.sender[0].id
    if(this.id == 'panelA' && sid == 'panelB' || this.id == 'panelB' && sid == 'panelA'){
        ui.item.find('svg').hide()
    }
}

function InitSortable(){
        $(".sortable").sortable({
            connectWith: ".sortable",
            helper: "clone",
            cursorAt: {left: 50, top: 50},
            receive: handleReceive
        }).disableSelection()
}

// parse the query string and return a dictionary.
var QueryString = function () {
  // This function is anonymous, is executed immediately and
  // the return value is assigned to QueryString!
  var query_string = {};
  var query = window.location.search.substring(1);
  var vars = query.split("&");
  for (var i=0;i<vars.length;i++) {
    var pair = vars[i].split("=");
        // If first entry with this name
    if (typeof query_string[pair[0]] === "undefined") {
      query_string[pair[0]] = decodeURIComponent(pair[1]);
        // If second entry with this name
    } else if (typeof query_string[pair[0]] === "string") {
      var arr = [ query_string[pair[0]],decodeURIComponent(pair[1]) ];
      query_string[pair[0]] = arr;
        // If third or later entry with this name
    } else {
      query_string[pair[0]].push(decodeURIComponent(pair[1]));
    }
  }
    return query_string;
}();

export function init(){
    let dataset = QueryString['dataset'];
    $.ajax({
        url: `../../../../../v1/query?q=select regex_replace(regex_replace(location, '/.*/', ''), '.jpg', '') from sample(${dataset},{rows:10})&format=table&rowNames=false&headers=false`,
    }).done((rows: Row[]) => {
        let s = rows2State(rows)
        let u = ui(s)
        document.body.appendChild(create(u))
        InitSortable()
    })
}

function HSVtoRGB(h: number, s: number, v: number) {
    let i = Math.floor(h * 6)
    let f = h * 6 - i
    let p = v * (1 - s)
    let q = v * (1 - f * s)
    let t = v * (1 - (1 - f) * s)
    var r: number, g: number, b : number
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    let he = (x: number) => ('0' + Math.round( x * 255).toString(16)).substr(-2)
    return `#${he(r)}${he(g)}${he(b)}`
}

$(init)

